import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { CommunityBoard } from "./CommunityBoard";
import { PostDetail } from "./PostDetail";
import { NewPostForm } from "./NewPostForm";

import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";

import { Info, Plus, Search as SearchIcon } from "lucide-react";

import type { Post, PostCategory, SortKey, NewPostDraft } from "../types/community";
import {
	fetchCommunityPosts,
	fetchCommunityPost,
	createCommunityPost,
	updateCommunityPost,
	deleteCommunityPost,
	likeCommunityPost,
	unlikeCommunityPost,
	createCommunityComment,
	deleteCommunityComment,
	uploadCommunityAttachments,
} from "../api/community";

type ViewMode = "list" | "detail" | "new";
type CategoryFilter = "all" | PostCategory;

const categoryLabel: Record<CategoryFilter, string> = {
	all: "전체",
	question: "질문",
	info: "정보",
	review: "후기",
	discussion: "토론",
};

function safeUserId() {
	return localStorage.getItem("userId") || "";
}
function isAuthedNow() {
	return !!localStorage.getItem("userId");
}

export function CommunityPage() {
	const navigate = useNavigate();

	const [viewMode, setViewMode] = useState<ViewMode>("list");
	const [searchQuery, setSearchQuery] = useState("");
	const [category, setCategory] = useState<CategoryFilter>("all");
	const [sortKey, setSortKey] = useState<SortKey>("latest");

	const [posts, setPosts] = useState<Post[]>([]);
	const [counts, setCounts] = useState<Record<"all" | PostCategory, number>>({
		all: 0, question: 0, info: 0, review: 0, discussion: 0,
	});

	const [listLoading, setListLoading] = useState(false);
	const [listError, setListError] = useState<string | null>(null);

	const [selectedPost, setSelectedPost] = useState<Post | null>(null);
	const [detailLoading, setDetailLoading] = useState(false);
	const [detailError, setDetailError] = useState<string | null>(null);

	const currentUserId = useMemo(() => safeUserId(), []);
	const authed = isAuthedNow();

	const goLogin = () => navigate("/login", { state: { from: "/community" } });

	const loadList = async (opts?: { q?: string; category?: CategoryFilter; sort?: SortKey }) => {
		setListLoading(true);
		setListError(null);

		try {
			const selectedCategory = opts?.category ?? category;

			const apiCategory = selectedCategory === "all" ? undefined : selectedCategory;

			const data = await fetchCommunityPosts({
				category: apiCategory,
				q: (opts?.q ?? searchQuery).trim() || undefined,
				sort: opts?.sort ?? sortKey,
				page: 1,
				size: 50,
			});

			setPosts(data.items);

			if (data.counts) {
				setCounts({ ...data.counts, all: data.counts.all });
			} else {
				// fallback: 현재 로드된 items 기준
				const base = { all: data.items.length, question: 0, info: 0, review: 0, discussion: 0 } as any;
				data.items.forEach((p) => { base[p.category] += 1; });
				setCounts(base);
			}
		} catch (e: any) {
			setListError(e?.message || "게시글 목록을 불러오지 못했습니다.");
		} finally {
			setListLoading(false);
		}
	};

	const loadDetail = async (postId: number) => {
		setDetailLoading(true);
		setDetailError(null);

		try {
			const data = await fetchCommunityPost(postId);

			// 안전하게 배열 기본값 세팅
			const fixed: Post = {
				...data,
				comments: data.comments ?? [],
				attachments: data.attachments ?? [],
			};

			setSelectedPost(fixed);
		} catch (e: any) {
			setDetailError(e?.message || "게시글을 불러오지 못했습니다.");
		} finally {
			setDetailLoading(false);
		}
	};

	useEffect(() => {
		// category/sort/search 변경 시 250ms 디바운스 로딩
		const t = window.setTimeout(() => {
			loadList();
		}, 250);

		return () => window.clearTimeout(t);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [category, sortKey, searchQuery]);

	const onSubmitSearch = (e: React.FormEvent) => {
		e.preventDefault();
		loadList({ q: searchQuery });
	};

	const openDetail = (post: Post) => {
		setViewMode("detail");
		setSelectedPost(null);
		loadDetail(post.id);
	};

	const backToList = () => {
		setSelectedPost(null);
		setViewMode("list");
	};

	const onClickWrite = () => {
		if (!authed) return goLogin();
		setViewMode("new");
	};

	const canEditSelected = useMemo(() => {
		if (!authed || !selectedPost) return false;
		if (!selectedPost.authorId) return false;
		return selectedPost.authorId === currentUserId;
	}, [authed, selectedPost, currentUserId]);

	const addPost = async (draft: NewPostDraft) => {
		if (!authed) return goLogin();

		try {
			// 첨부가 있으면 먼저 업로드(정의서에 API 추가 필요)
			let attachmentIds: string[] | undefined;
			if (draft.files.length > 0) {
				const uploaded = await uploadCommunityAttachments(draft.files);
				attachmentIds = uploaded.map((a) => a.id);
			}

			const created = await createCommunityPost({
				title: draft.title,
				content: draft.content,
				category: draft.category,
				attachmentIds,
			});

			setViewMode("detail");
			setSelectedPost(null);
			await loadList();
			await loadDetail(created.id);
		} catch (e: any) {
			alert(e?.message || "게시글 작성에 실패했습니다.");
		}
	};

	const addComment = async (postId: number, content: string) => {
		if (!authed) return goLogin();

		try {
			const created = await createCommunityComment(postId, content);

			setSelectedPost((prev) => {
				if (!prev || prev.id !== postId) return prev;
				const nextComments = [...(prev.comments ?? []), created];
				return { ...prev, comments: nextComments, commentCount: nextComments.length };
			});

			setPosts((prev) =>
				prev.map((p) =>
					p.id === postId
						? { ...p, commentCount: (p.commentCount ?? 0) + 1 }
						: p,
				),
			);
		} catch (e: any) {
			alert(e?.message || "댓글 작성에 실패했습니다.");
		}
	};

	const updatePost = async (postId: string, patch: Partial<Pick<Post, "title" | "content" | "category">>) => {
		if (!authed) return goLogin();

		try {
			const updated = await updateCommunityPost(postId, {
				title: patch.title,
				content: patch.content,
				category: patch.category,
			});

			const fixed: Post = { ...updated, comments: updated.comments ?? [], attachments: updated.attachments ?? [] };
			setSelectedPost((prev) => (prev?.id === postId ? fixed : prev));
			setPosts((prev) => prev.map((p) => (p.id === postId ? { ...p, ...fixed } : p)));
		} catch (e: any) {
			alert(e?.message || "게시글 수정에 실패했습니다.");
		}
	};

	const deletePost = async (postId: number) => {
		if (!authed) return goLogin();

		try {
			await deleteCommunityPost(postId);
			backToList();
			await loadList();
		} catch (e: any) {
			alert(e?.message || "게시글 삭제에 실패했습니다.");
		}
	};

	const togglePostLike = async (postId: string) => {
		if (!authed) return goLogin();

		const target = selectedPost?.id === postId ? selectedPost : posts.find((p) => p.id === postId);
		const liked = !!target?.likedByMe;

		try {
			if (liked) await unlikeCommunityPost(postId);
			else await likeCommunityPost(postId);

			const delta = liked ? -1 : 1;

			setPosts((prev) =>
				prev.map((p) =>
					p.id === postId
						? { ...p, likedByMe: !liked, likes: Math.max(0, p.likes + delta) }
						: p,
				),
			);

			setSelectedPost((prev) => {
				if (!prev || prev.id !== postId) return prev;
				return { ...prev, likedByMe: !liked, likes: Math.max(0, prev.likes + delta) };
			});
		} catch (e: any) {
			alert(e?.message || "좋아요 처리에 실패했습니다.");
		}
	};

	const deleteComment = async (postId: number, commentId: string) => {
		if (!authed) return goLogin();

		try {
			await deleteCommunityComment(postId, commentId);

			setSelectedPost((prev) => {
				if (!prev || prev.id !== postId) return prev;
				const next = (prev.comments ?? []).filter((c) => c.id !== commentId);
				return { ...prev, comments: next, commentCount: next.length };
			});

			setPosts((prev) =>
				prev.map((p) =>
					p.id === postId
						? { ...p, commentCount: Math.max(0, (p.commentCount ?? 0) - 1) }
						: p,
				),
			);
		} catch (e: any) {
			alert(e?.message || "댓글 삭제에 실패했습니다.");
		}
	};

	return (
		<div className="space-y-4">
			{viewMode === "list" && (
				<Card>
					<CardHeader className="space-y-1">
						<div className="flex items-start justify-between gap-4">
							<div>
								<CardTitle className="text-xl">커뮤니티</CardTitle>
								<CardDescription>
									입찰 실무 노하우/질문/정보를 빠르게 공유하고 검색하세요.
								</CardDescription>
							</div>

							<div className="shrink-0">
								<Button
									variant={authed ? "default" : "outline"}
									onClick={onClickWrite}
									className="gap-2"
								>
									{authed ? <Plus className="h-4 w-4" /> : null}
									{authed ? "글쓰기" : "로그인 후 글쓰기"}
								</Button>
							</div>
						</div>
					</CardHeader>

					<CardContent className="space-y-3">
						<Tabs value={category} onValueChange={(v) => setCategory(v as CategoryFilter)}>
							<div className="flex items-center justify-between gap-3">
								<TabsList className="w-fit justify-start rounded-lg">
									<TabsTrigger value="all" className="flex-none px-3 rounded-md text-[13px]">
										전체 <span className="ml-1 text-xs opacity-70">{counts.all}</span>
									</TabsTrigger>
									<TabsTrigger value="question" className="flex-none px-3 rounded-md text-[13px]">
										질문 <span className="ml-1 text-xs opacity-70">{counts.question}</span>
									</TabsTrigger>
									<TabsTrigger value="info" className="flex-none px-3 rounded-md text-[13px]">
										정보 <span className="ml-1 text-xs opacity-70">{counts.info}</span>
									</TabsTrigger>
									<TabsTrigger value="review" className="flex-none px-3 rounded-md text-[13px]">
										후기 <span className="ml-1 text-xs opacity-70">{counts.review}</span>
									</TabsTrigger>
									<TabsTrigger value="discussion" className="flex-none px-3 rounded-md text-[13px]">
										토론 <span className="ml-1 text-xs opacity-70">{counts.discussion}</span>
									</TabsTrigger>
								</TabsList>

								<div className="hidden md:block text-xs text-gray-500 tabular-nums">
									{categoryLabel[category]} · {posts.length}건
								</div>
							</div>
						</Tabs>

						<form onSubmit={onSubmitSearch} className="flex flex-col md:flex-row md:items-center gap-2">
							<div className="relative flex-1">
								<SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
								<Input
									value={searchQuery}
									onChange={(e) => setSearchQuery(e.target.value)}
									placeholder="검색: 제목/내용/작성자"
									className="pl-9"
								/>
							</div>

							<div className="flex items-center gap-2">
								<Button type="submit" className="whitespace-nowrap">검색하기</Button>

								<Select value={sortKey} onValueChange={(v) => setSortKey(v as SortKey)}>
									<SelectTrigger className="w-[140px]">
										<SelectValue placeholder="정렬" />
									</SelectTrigger>
									<SelectContent>
										<SelectItem value="latest">최신순</SelectItem>
										<SelectItem value="popular">인기순</SelectItem>
										<SelectItem value="views">조회순</SelectItem>
										<SelectItem value="comments">댓글순</SelectItem>
									</SelectContent>
								</Select>
							</div>
						</form>

						{!authed && (
							<Alert className="bg-slate-50">
								<Info />
								<AlertTitle>게스트 모드</AlertTitle>
								<AlertDescription>글쓰기/좋아요/댓글은 로그인 후 이용할 수 있습니다.</AlertDescription>
							</Alert>
						)}

						{listError && (
							<div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
								{listError}
							</div>
						)}
					</CardContent>
				</Card>
			)}

			{viewMode === "list" && (
				listLoading ? (
					<div className="text-sm text-gray-500 py-8 text-center">목록을 불러오는 중...</div>
				) : (
					<CommunityBoard posts={posts} onSelectPost={openDetail} />
				)
			)}

			{viewMode === "detail" && (
				detailLoading ? (
					<div className="text-sm text-gray-500 py-8 text-center">게시글을 불러오는 중...</div>
				) : detailError ? (
					<div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
						{detailError}
					</div>
				) : selectedPost ? (
					<PostDetail
						post={selectedPost}
						onBack={backToList}
						onAddComment={addComment}
						onUpdatePost={updatePost}
						onDeletePost={deletePost}
						onToggleLike={togglePostLike}
						onDeleteComment={deleteComment}
						canEdit={canEditSelected}
						canInteract={authed}
						onRequireAuth={goLogin}
						currentUserId={currentUserId}
					/>
				) : null
			)}

			{viewMode === "new" && (
				<NewPostForm onSubmit={addPost} onCancel={() => setViewMode("list")} />
			)}
		</div>
	);
}
