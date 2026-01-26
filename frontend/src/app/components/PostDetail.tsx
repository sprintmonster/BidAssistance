import { useEffect, useMemo, useState } from "react";
import {
	ArrowLeft,
	Download,
	Eye,
	Image as ImageIcon,
	FileText as FileTextIcon,
	MessageSquare,
	Send,
	ThumbsUp,
} from "lucide-react";

import type { Post, PostCategory } from "../types/community";
import { mask_name } from "../utils/masking";

interface PostDetailProps {
	post: Post;
	onBack: () => void;

	onAddComment: (postId: number, content: string) => void;
	onUpdatePost: (
		postId: number,
		patch: Partial<Pick<Post, "title" | "content" | "category">>,
	) => void;
	onDeletePost: (postId: number) => void;
	onToggleLike: (postId: number) => void;
	onDeleteComment: (postId: number, commentId: string) => void;

	canEdit?: boolean;
	canInteract?: boolean;
	onRequireAuth?: () => void;
	currentUserId?: string;
}

const category_labels: Record<PostCategory, string> = {
	question: "질문",
	info: "정보",
	review: "후기",
	discussion: "토론",
};

const category_colors: Record<PostCategory, string> = {
	question: "bg-blue-100 text-blue-800",
	info: "bg-green-100 text-green-800",
	review: "bg-purple-100 text-purple-800",
	discussion: "bg-orange-100 text-orange-800",
};

function safe_local_name() {
	try {
		return localStorage.getItem("name") || "";
	} catch {
		return "";
	}
}

export function PostDetail({
	post,
	onBack,
	onAddComment,
	onUpdatePost,
	onDeletePost,
	onToggleLike,
	onDeleteComment,
	canEdit = false,
	canInteract = false,
	onRequireAuth,
	currentUserId,
}: PostDetailProps) {
	const comments = useMemo(() => post.comments ?? [], [post.comments]);
	const attachments = useMemo(() => post.attachments ?? [], [post.attachments]);
	const comment_count = post.commentCount ?? comments.length;

	const author_name = (post as any).authorName ?? (post as any).author ?? "—";

	const [comment_text, set_comment_text] = useState("");
	const [is_editing, set_is_editing] = useState(false);

	const [edit_title, set_edit_title] = useState(post.title ?? "");
	const [edit_content, set_edit_content] = useState(post.content ?? "");
	const [edit_category, set_edit_category] = useState<PostCategory>(post.category);

	useEffect(() => {
		if (!is_editing) {
			set_edit_title(post.title ?? "");
			set_edit_content(post.content ?? "");
			set_edit_category(post.category);
		}
	}, [post.id, is_editing, post.title, post.content, post.category]);

	const submit_comment = () => {
		if (!canInteract) {
			onRequireAuth?.();
			return;
		}
		const text = comment_text.trim();
		if (!text) return;

		onAddComment(post.id, text);
		set_comment_text("");
	};

	const start_edit = () => {
		if (!canEdit) {
			onRequireAuth?.();
			return;
		}
		set_is_editing(true);
		set_edit_title(post.title ?? "");
		set_edit_content(post.content ?? "");
		set_edit_category(post.category);
	};

	const cancel_edit = () => {
		set_is_editing(false);
		set_edit_title(post.title ?? "");
		set_edit_content(post.content ?? "");
		set_edit_category(post.category);
	};

	const save_edit = () => {
		if (!canEdit) {
			onRequireAuth?.();
			return;
		}

		const t = edit_title.trim();
		const c = edit_content.trim();
		if (!t || !c) return;

		onUpdatePost(post.id, { title: t, content: c, category: edit_category });
		set_is_editing(false);
	};

	const delete_post = () => {
		if (!canEdit) {
			onRequireAuth?.();
			return;
		}
		const ok = window.confirm("이 게시글을 삭제할까요? 삭제 후 복구할 수 없습니다.");
		if (!ok) return;
		onDeletePost(post.id);
	};

	const toggle_like = () => {
		if (!canInteract) {
			onRequireAuth?.();
			return;
		}
		onToggleLike(post.id);
	};

	const local_name = safe_local_name();

	return (
		<div className="space-y-6">
			<button
				onClick={onBack}
				className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
			>
				<ArrowLeft className="w-5 h-5" />
				<span>목록으로</span>
			</button>

			<div className="bg-white rounded-lg border border-gray-200 p-8">
				<div className="mb-4">
					{is_editing ? (
						<select
							value={edit_category}
							onChange={(e) => set_edit_category(e.target.value as PostCategory)}
							className="px-3 py-2 border rounded-lg text-sm"
						>
							<option value="question">질문</option>
							<option value="info">정보</option>
							<option value="review">후기</option>
							<option value="discussion">토론</option>
						</select>
					) : (
						<span
							className={`px-3 py-1 rounded text-sm font-medium ${category_colors[post.category]}`}
						>
							{category_labels[post.category]}
						</span>
					)}
				</div>

				<div className="flex items-start justify-between gap-3 mb-4">
					<div className="min-w-0 flex-1">
						{is_editing ? (
							<input
								value={edit_title}
								onChange={(e) => set_edit_title(e.target.value)}
								className="w-full text-2xl font-bold border rounded-lg px-3 py-2"
							/>
						) : (
							<h1 className="text-3xl font-bold text-gray-900 truncate">{post.title}</h1>
						)}
					</div>

					<div className="shrink-0 flex gap-2">
						{canEdit && !is_editing && (
							<button
								type="button"
								onClick={delete_post}
								className="px-3 py-2 rounded-lg border border-red-300 text-red-600 hover:bg-red-50"
							>
								삭제
							</button>
						)}

						{canEdit &&
							(is_editing ? (
								<>
									<button
										type="button"
										onClick={cancel_edit}
										className="px-3 py-2 rounded-lg border"
									>
										취소
									</button>
									<button
										type="button"
										onClick={save_edit}
										className="px-3 py-2 rounded-lg bg-blue-600 text-white"
									>
										저장
									</button>
								</>
							) : (
								<button
									type="button"
									onClick={start_edit}
									className="px-3 py-2 rounded-lg border"
								>
									수정
								</button>
							))}
					</div>
				</div>

				<div className="flex items-center gap-4 text-sm text-gray-500 mb-6 pb-6 border-b border-gray-200">
					<span className="font-medium text-gray-700">{mask_name(author_name)}</span>
					<span>·</span>
					<span>{post.createdAt}</span>
					<span>·</span>
					<div className="flex items-center gap-1">
						<Eye className="w-4 h-4" />
						<span>{post.views}</span>
					</div>
				</div>

				<div className="prose max-w-none mb-6">
					{is_editing ? (
						<textarea
							value={edit_content}
							onChange={(e) => set_edit_content(e.target.value)}
							className="w-full border rounded-lg px-3 py-2 min-h-[240px]"
						/>
					) : (
						<p className="text-gray-700 leading-relaxed whitespace-pre-wrap">{post.content ?? ""}</p>
					)}
				</div>

				{attachments.length > 0 && (
					<div className="mb-6">
						<div className="text-sm font-medium text-gray-700 mb-2">첨부파일</div>

						<div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-3">
							{attachments
								.filter((a) => a.isImage)
								.map((a) => (
									<button
										key={a.id}
										type="button"
										onClick={() => window.open(a.url, "_blank")}
										className="group border rounded-lg overflow-hidden bg-gray-50 hover:shadow-sm transition text-left"
										title={a.name}
									>
										<img src={a.url} alt={a.name} className="w-full h-32 object-cover" />
										<div className="p-2 text-xs text-gray-700 truncate">{a.name}</div>
									</button>
								))}
						</div>

						<div className="space-y-2">
							{attachments.map((a) => (
								<div
									key={a.id}
									className="flex items-center justify-between gap-3 border rounded-lg px-3 py-2"
								>
									<div className="flex items-center gap-2 min-w-0">
										{a.isImage ? (
											<ImageIcon className="w-4 h-4 text-gray-500" />
										) : (
											<FileTextIcon className="w-4 h-4 text-gray-500" />
										)}
										<div className="min-w-0">
											<div className="text-sm text-gray-900 truncate">{a.name}</div>
											<div className="text-xs text-gray-500">
												{Math.round((a.size ?? 0) / 1024)} KB
											</div>
										</div>
									</div>

									<a
										href={a.url}
										download={a.name}
										className="shrink-0 inline-flex items-center gap-1 px-3 py-1.5 text-sm rounded-md border hover:bg-gray-50"
										onClick={(e) => e.stopPropagation()}
									>
										<Download className="w-4 h-4" />
										다운로드
									</a>
								</div>
							))}
						</div>
					</div>
				)}

				<div className="flex items-center gap-4 pt-6 border-t border-gray-200">
					<button
						type="button"
						onClick={toggle_like}
						className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
							post.likedByMe
								? "bg-blue-600 text-white hover:bg-blue-700"
								: "bg-blue-50 text-blue-600 hover:bg-blue-100"
						} ${canInteract ? "" : "opacity-60"}`}
					>
						<ThumbsUp className="w-5 h-5" />
						<span>{canInteract ? `좋아요 ${post.likes}` : `로그인 후 좋아요 (${post.likes})`}</span>
					</button>
				</div>
			</div>

			<div className="bg-white rounded-lg border border-gray-200 p-8">
				<div className="flex items-center gap-2 mb-6">
					<MessageSquare className="w-5 h-5 text-gray-700" />
					<h2 className="text-xl font-bold text-gray-900">댓글 {comment_count}</h2>
				</div>

				<div className="mb-8">
					<textarea
						value={comment_text}
						onChange={(e) => set_comment_text(e.target.value)}
						placeholder={canInteract ? "댓글을 입력하세요..." : "로그인 후 댓글을 작성할 수 있습니다."}
						disabled={!canInteract}
						className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none disabled:bg-gray-50 disabled:text-gray-400"
						rows={4}
					/>
					<div className="flex justify-end mt-2">
						<button
							type="button"
							onClick={submit_comment}
							disabled={!canInteract || !comment_text.trim()}
							className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
						>
							<Send className="w-4 h-4" />
							댓글 작성
						</button>
					</div>
				</div>

				<div className="space-y-6">
					{comments.map((comment: any) => {
						const comment_author_name = comment.authorName ?? comment.author ?? "—";
						const comment_author_id = comment.authorId as string | undefined;

						const can_delete_this_comment =
							!!canInteract &&
							(!!canEdit ||
								(!!comment_author_id &&
									!!currentUserId &&
									comment_author_id === currentUserId) ||
								(!!local_name && comment_author_name === local_name));

						return (
							<div
								key={comment.id}
								className="border-b border-gray-100 pb-6 last:border-b-0 last:pb-0"
							>
								<div className="flex items-center justify-between mb-2">
									<div className="flex items-center gap-3">
										<span className="font-medium text-gray-900">
											{mask_name(comment_author_name)}
										</span>
										<span className="text-sm text-gray-500">{comment.createdAt}</span>
									</div>

									{can_delete_this_comment && (
										<button
											type="button"
											onClick={() => {
												const ok = window.confirm("이 댓글을 삭제할까요?");
												if (!ok) return;
												onDeleteComment(post.id, String(comment.id));
											}}
											className="text-sm text-gray-400 hover:text-red-600"
										>
											삭제
										</button>
									)}
								</div>

								<p className="text-gray-700 mb-3 whitespace-pre-wrap">{comment.content}</p>

								<button
									type="button"
									className="flex items-center gap-1 text-sm text-gray-500 hover:text-blue-600 transition-colors"
									onClick={() => {
										if (!canInteract) onRequireAuth?.();
									}}
								>
									<ThumbsUp className="w-4 h-4" />
									<span>{comment.likes ?? 0}</span>
								</button>
							</div>
						);
					})}

					{comments.length === 0 && (
						<p className="text-center text-gray-500 py-8">첫 번째 댓글을 작성해보세요!</p>
					)}
				</div>
			</div>
		</div>
	);
}
