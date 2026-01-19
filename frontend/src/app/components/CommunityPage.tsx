import { useMemo, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { CommunityBoard } from "./CommunityBoard";
import { PostDetail } from "./PostDetail";
import { NewPostForm } from "./NewPostForm";

import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Card, CardContent, CardHeader } from "./ui/card";
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "./ui/select";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";

import { Info, Plus, Search as SearchIcon } from "lucide-react";

export type Comment = {
	id: string;
	author: string;
	content: string;
	createdAt: string;
	likes: number;
};

export type Attachment = {
	id: string;
	name: string;
	type: string;
	url: string;
	size: number;
	isImage: boolean;
};

export type Post = {
	id: string;
	title: string;
	content: string;
	category: "question" | "info" | "review" | "discussion";
	author: string;
	createdAt: string; // YYYY-MM-DD
	views: number;
	likes: number;
	likedByMe: boolean;
	comments: Comment[];
	attachments: Attachment[];
};

type ViewMode = "list" | "detail" | "new";
type CategoryFilter = "all" | Post["category"];
type SortKey = "latest" | "popular" | "views" | "comments";

const categoryLabel: Record<CategoryFilter, string> = {
	all: "전체",
	question: "질문",
	info: "정보",
	review: "후기",
	discussion: "토론",
};

export function CommunityPage() {
	const navigate = useNavigate();
	const isAuthed = useMemo(() => !!localStorage.getItem("accessToken"), []);

	const [viewMode, setViewMode] = useState<ViewMode>("list");
	const [searchQuery, setSearchQuery] = useState("");
	const [category, setCategory] = useState<CategoryFilter>("all");
	const [sortKey, setSortKey] = useState<SortKey>("latest");
	const [selectedPost, setSelectedPost] = useState<Post | null>(null);

	const [posts, setPosts] = useState<Post[]>([
		{
			id: "p1",
			title: "입찰 서류 준비할 때 체크리스트 있을까요?",
			content:
				"처음 입찰 참여하는데 필수로 챙겨야 할 서류/실수 방지 팁 공유 부탁드립니다.",
			category: "question",
			author: "사용자A",
			createdAt: "2026-01-12",
			views: 12,
			likes: 3,
			likedByMe: false,
			comments: [],
			attachments: [],
		},
		{
			id: "p2",
			title: "마감 임박 공고 우선순위 정하는 기준 공유",
			content:
				"마감 임박 공고가 많을 때, 금액/지역/경쟁률 기반으로 우선순위 정하는 방법 공유합니다.",
			category: "info",
			author: "사용자B",
			createdAt: "2026-01-11",
			views: 30,
			likes: 8,
			likedByMe: false,
			comments: [],
			attachments: [],
		},
	]);

	const counts = useMemo(() => {
		const base = { all: posts.length, question: 0, info: 0, review: 0, discussion: 0 };
		posts.forEach((p) => {
			base[p.category] += 1;
		});
		return base;
	}, [posts]);

	const filteredSorted = useMemo(() => {
		const q = searchQuery.trim().toLowerCase();

		let list = posts;

		if (category !== "all") {
			list = list.filter((p) => p.category === category);
		}

		if (q) {
			list = list.filter(
				(p) =>
					p.title.toLowerCase().includes(q) ||
					p.content.toLowerCase().includes(q) ||
					p.author.toLowerCase().includes(q),
			);
		}

		const toTime = (d: string) => new Date(d).getTime() || 0;

		const sorted = [...list].sort((a, b) => {
			if (sortKey === "popular") {
				if (b.likes !== a.likes) return b.likes - a.likes;
				return toTime(b.createdAt) - toTime(a.createdAt);
			}
			if (sortKey === "views") {
				if (b.views !== a.views) return b.views - a.views;
				return toTime(b.createdAt) - toTime(a.createdAt);
			}
			if (sortKey === "comments") {
				if (b.comments.length !== a.comments.length)
					return b.comments.length - a.comments.length;
				return toTime(b.createdAt) - toTime(a.createdAt);
			}
			return toTime(b.createdAt) - toTime(a.createdAt);
		});

		return sorted;
	}, [posts, searchQuery, category, sortKey]);

	const goLogin = () => {
		navigate("/login", { state: { from: "/community" } });
	};

	const openDetail = (post: Post) => {
		const nextPost = { ...post, views: post.views + 1 };
		setPosts((prev) => prev.map((x) => (x.id === post.id ? nextPost : x)));
		setSelectedPost(nextPost);
		setViewMode("detail");
	};

	const backToList = () => {
		setSelectedPost(null);
		setViewMode("list");
	};

	const addPost = (
		newPost: Omit<Post, "id" | "createdAt" | "views" | "likes" | "comments">,
	) => {
		const createdAt = new Date().toISOString().slice(0, 10);
		const post: Post = {
			...newPost,
			id: `p_${Date.now()}`,
			createdAt,
			views: 0,
			likes: 0,
			comments: [],
		};
		setPosts((prev) => [post, ...prev]);
		setViewMode("list");
	};

	const addComment = (postId: string, content: string) => {
		const createdAt = new Date().toISOString().slice(0, 10);

		setPosts((prev) =>
			prev.map((p) => {
				if (p.id !== postId) return p;
				return {
					...p,
					comments: [
						...p.comments,
						{
							id: `c_${Date.now()}`,
							author: "사용자",
							content,
							createdAt,
							likes: 0,
						},
					],
				};
			}),
		);

		setSelectedPost((prev) => {
			if (!prev || prev.id !== postId) return prev;
			return {
				...prev,
				comments: [
					...prev.comments,
					{
						id: `c_${Date.now()}`,
						author: "사용자",
						content,
						createdAt,
						likes: 0,
					},
				],
			};
		});
	};

	const updatePost = (
		postId: string,
		patch: Partial<Pick<Post, "title" | "content" | "category" | "attachments">>,
	) => {
		setPosts((prev) => prev.map((p) => (p.id === postId ? { ...p, ...patch } : p)));
		setSelectedPost((prev) => {
			if (!prev || prev.id !== postId) return prev;
			return { ...prev, ...patch };
		});
	};

	const deletePost = (postId: string) => {
		setPosts((prev) => prev.filter((p) => p.id !== postId));
		setSelectedPost((prev) => (prev?.id === postId ? null : prev));
		if (selectedPost?.id === postId) setViewMode("list");
	};

	const togglePostLike = (postId: string) => {
		setPosts((prev) =>
			prev.map((p) => {
				if (p.id !== postId) return p;
				const nextLiked = !p.likedByMe;
				return {
					...p,
					likedByMe: nextLiked,
					likes: Math.max(0, p.likes + (nextLiked ? 1 : -1)),
				};
			}),
		);
		setSelectedPost((prev) => {
			if (!prev || prev.id !== postId) return prev;
			const nextLiked = !prev.likedByMe;
			return {
				...prev,
				likedByMe: nextLiked,
				likes: Math.max(0, prev.likes + (nextLiked ? 1 : -1)),
			};
		});
	};

	const deleteComment = (postId: string, commentId: string) => {
		setPosts((prev) =>
			prev.map((p) => {
				if (p.id !== postId) return p;
				return { ...p, comments: p.comments.filter((c) => c.id !== commentId) };
			}),
		);
		setSelectedPost((prev) => {
			if (!prev || prev.id !== postId) return prev;
			return { ...prev, comments: prev.comments.filter((c) => c.id !== commentId) };
		});
	};

	useEffect(() => {
		window.scrollTo({ top: 0, left: 0, behavior: "auto" });
	}, [viewMode, selectedPost?.id]);

	const onClickWrite = () => {
		if (!isAuthed) return goLogin();
		setViewMode("new");
	};

	const onSubmitSearch = (e: React.FormEvent) => {
		e.preventDefault();
		// 현재는 로컬 필터링이므로 별도 동작 없음.
		// 추후 서버 검색 붙이면 여기에서 fetch 트리거하면 됨.
	};

	return (
		<div className="space-y-4">
			{viewMode === "list" && (
				<Card className="border bg-white">
					<CardHeader className="pb-3">
						<div className="flex items-start justify-between gap-4">
							<div>
								<div className="text-lg font-semibold text-gray-900">커뮤니티</div>
								<div className="mt-1 text-sm text-gray-600">
									입찰 실무 노하우/질문/정보를 빠르게 공유하고 검색하세요.
								</div>
							</div>

							<div className="shrink-0">
								<Button
									variant={isAuthed ? "default" : "outline"}
									onClick={onClickWrite}
									className="gap-2"
								>
									{isAuthed ? <Plus className="h-4 w-4" /> : null}
									{isAuthed ? "글쓰기" : "로그인 후 글쓰기"}
								</Button>
							</div>
						</div>
					</CardHeader>

					<CardContent className="pt-0 space-y-3">
						<Tabs
							value={category}
							onValueChange={(v) => setCategory(v as CategoryFilter)}
						>
							<div className="flex items-center justify-between gap-3">
								<TabsList className="w-fit justify-start rounded-lg">
									<TabsTrigger
										value="all"
										className="flex-none px-3 rounded-md text-[13px]"
									>
										전체{" "}
										<span className="ml-1 text-xs opacity-70">{counts.all}</span>
									</TabsTrigger>
									<TabsTrigger
										value="question"
										className="flex-none px-3 rounded-md text-[13px]"
									>
										질문{" "}
										<span className="ml-1 text-xs opacity-70">{counts.question}</span>
									</TabsTrigger>
									<TabsTrigger
										value="info"
										className="flex-none px-3 rounded-md text-[13px]"
									>
										정보{" "}
										<span className="ml-1 text-xs opacity-70">{counts.info}</span>
									</TabsTrigger>
									<TabsTrigger
										value="review"
										className="flex-none px-3 rounded-md text-[13px]"
									>
										후기{" "}
										<span className="ml-1 text-xs opacity-70">{counts.review}</span>
									</TabsTrigger>
									<TabsTrigger
										value="discussion"
										className="flex-none px-3 rounded-md text-[13px]"
									>
										토론{" "}
										<span className="ml-1 text-xs opacity-70">
											{counts.discussion}
										</span>
									</TabsTrigger>
								</TabsList>

								<div className="hidden md:block text-xs text-gray-500 tabular-nums">
									{categoryLabel[category]} · {filteredSorted.length}건
								</div>
							</div>
						</Tabs>

						{/* ✅ 검색 버튼 포함: form으로 감싸고 버튼 추가 */}
						<form
							onSubmit={onSubmitSearch}
							className="flex flex-col md:flex-row md:items-center gap-2"
						>
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
								<Button type="submit" className="whitespace-nowrap">
									검색하기
								</Button>

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

						{!isAuthed && (
							<Alert className="bg-slate-50">
								<Info />
								<AlertTitle>게스트 모드</AlertTitle>
								<AlertDescription>
									글쓰기/좋아요/댓글은 로그인 후 이용할 수 있습니다.
								</AlertDescription>
							</Alert>
						)}
					</CardContent>
				</Card>
			)}

			{viewMode === "list" && (
				<CommunityBoard posts={filteredSorted} onSelectPost={openDetail} />
			)}

			{viewMode === "detail" && selectedPost && (
				<PostDetail
					post={selectedPost}
					onBack={backToList}
					onAddComment={addComment}
					onUpdatePost={updatePost}
					onDeletePost={deletePost}
					onToggleLike={togglePostLike}
					onDeleteComment={deleteComment}
				/>
			)}

			{viewMode === "new" && (
				<NewPostForm onSubmit={addPost} onCancel={() => setViewMode("list")} />
			)}
		</div>
	);
}
