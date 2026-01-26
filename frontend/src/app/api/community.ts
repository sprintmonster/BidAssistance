import { api } from "./client";
import type {
	ApiResponse,
	Post,
	PostListData,
	SortKey,
	PostCategory,
	Comment,
	Attachment,
	Id,
} from "../types/community";

function qs(params: Record<string, string | number | undefined>) {
	const sp = new URLSearchParams();
	Object.entries(params).forEach(([k, v]) => {
		if (v === undefined || v === "") return;
		sp.set(k, String(v));
	});
	const s = sp.toString();
	return s ? `?${s}` : "";
}

function unwrap<T>(res: ApiResponse<T>): T {
	if (res.status !== "success") throw new Error(res.message || "요청에 실패했습니다.");
	return res.data;
}

/** 서버 필드(postId/commentId/attachmentId)를 프론트 필드(id)로 정규화 */
function normAttachment(a: any): Attachment {
	return {
		id: a.id ?? a.attachmentId,
		name: a.name,
		type: a.type,
		url: a.url,
		size: Number(a.size ?? 0),
		isImage: Boolean(a.isImage),
	};
}

function normComment(c: any): Comment {
	return {
		id: c.id ?? c.commentId,
		authorId: c.authorId,
		authorName: c.authorName ?? c.author ?? "—",
		content: c.content ?? "",
		createdAt: String(c.createdAt ?? ""),
		likes: Number(c.likes ?? 0),
		likedByMe: Boolean(c.likedByMe),
	};
}

function normPost(p: any): Post {
	return {
		id: p.id ?? p.postId,
		title: p.title ?? "",
		content: p.content,
		contentPreview: p.contentPreview,
		category: p.category,
		authorId: p.authorId,
		authorName: p.authorName ?? p.author ?? "—",
		createdAt: String(p.createdAt ?? ""),
		views: Number(p.views ?? 0),
		likes: Number(p.likes ?? 0),
		likedByMe: Boolean(p.likedByMe),
		commentCount: p.commentCount ?? p.commentsCount ?? p.comment_count,
		attachmentCount: p.attachmentCount ?? p.attachmentsCount ?? p.attachment_count,
		attachments: Array.isArray(p.attachments) ? p.attachments.map(normAttachment) : [],
		comments: Array.isArray(p.comments) ? p.comments.map(normComment) : [],
	};
}

function normPostList(d: any): PostListData {
	const itemsRaw = d.items ?? d.content ?? [];
	return {
		items: Array.isArray(itemsRaw) ? itemsRaw.map(normPost) : [],
		page: Number(d.page ?? 1),
		size: Number(d.size ?? 10),
		total: Number(d.total ?? d.totalElements ?? 0),
		counts: d.counts,
	};
}

/** (정의서) GET /api/board/posts */
export function fetchCommunityPosts(opts: {
	category?: PostCategory;
	q?: string;
	sort?: SortKey;
	page?: number;
	size?: number;
}) {
	return api<ApiResponse<any>>(
		`/board/posts${qs({
			category: opts.category,
			q: opts.q,
			sort: opts.sort,
			page: opts.page,
			size: opts.size,
		})}`,
	).then(unwrap).then(normPostList);
}

/** (정의서) GET /api/community/posts/{postId} */
export function fetchCommunityPost(postId: Id) {
	return api<ApiResponse<any>>(`/community/posts/${postId}`).then(unwrap).then(normPost);
}

/** (정의서) POST /api/community/posts */
export function createCommunityPost(payload: {
	title: string;
	content: string;
	category: PostCategory;
	attachmentIds?: string[];
}) {
	return api<ApiResponse<any>>("/community/posts", {
		method: "POST",
		body: JSON.stringify(payload),
	}).then(unwrap).then(normPost);
}

/** (정의서) PATCH /api/community/posts/{postId} */
export function updateCommunityPost(
	postId: Id,
	payload: Partial<{
		title: string;
		content: string;
		category: PostCategory;
		attachmentIds: string[];
	}>,
) {
	return api<ApiResponse<any>>(`/community/posts/${postId}`, {
		method: "PATCH",
		body: JSON.stringify(payload),
	}).then(unwrap).then(normPost);
}

/** (정의서) DELETE /api/community/posts/{postId} */
export function deleteCommunityPost(postId: Id) {
	return api<ApiResponse<{ message?: string }>>(`/community/posts/${postId}`, {
		method: "DELETE",
	}).then(unwrap);
}

/** (정의서) POST /api/board/posts/{postId}/like */
export function likeCommunityPost(postId: Id) {
	return api<ApiResponse<any>>(`/board/posts/${postId}/like`, { method: "POST" }).then(unwrap);
}

/** (정의서) POST /api/board/posts/{postId}/dislike */
export function unlikeCommunityPost(postId: Id) {
	return api<ApiResponse<any>>(`/board/posts/${postId}/dislike`, { method: "POST" }).then(unwrap);
}

/** (정의서) POST /api/community/posts/{postId}/comments */
export function createCommunityComment(postId: Id, content: string) {
	return api<ApiResponse<any>>(`/community/posts/${postId}/comments`, {
		method: "POST",
		body: JSON.stringify({ content }),
	}).then(unwrap).then(normComment);
}

/** (정의서) DELETE /api/community/posts/{postId}/comments/{commentId} */
export function deleteCommunityComment(postId: Id, commentId: Id) {
	return api<ApiResponse<{ message?: string }>>(
		`/community/posts/${postId}/comments/${commentId}`,
		{ method: "DELETE" },
	).then(unwrap);
}

/** (정의서) POST /api/board/attachments (multipart/form-data: files) */
export async function uploadCommunityAttachments(files: File[]) {
	const fd = new FormData();
	files.forEach((f) => fd.append("files", f));

	const res = await api<ApiResponse<any>>("/board/attachments", {
		method: "POST",
		body: fd,
	});

	const data = unwrap(res);
	const arr = Array.isArray(data) ? data : data.attachments ?? data.items ?? [];
	return (Array.isArray(arr) ? arr : []).map(normAttachment);
}
