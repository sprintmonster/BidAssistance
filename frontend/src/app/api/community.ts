import { api } from "./client";
import type {
	ApiResponse,
	Post,
	PostListData,
	SortKey,
	PostCategory,
	Comment,
	Attachment,
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

export function fetchCommunityPosts(opts: {
	category?: PostCategory;
	q?: string;
	sort?: SortKey;
	page?: number;
	size?: number;
}) {
	return api<ApiResponse<PostListData>>(
		`/community/posts${qs({
			category: opts.category,
			q: opts.q,
			sort: opts.sort,
			page: opts.page,
			size: opts.size,
		})}`,
	).then(unwrap);
}

export function fetchCommunityPost(postId: number) {
	return api<ApiResponse<Post>>(`/community/posts/${postId}`).then(unwrap);
}

export function createCommunityPost(payload: {
	title: string;
	content: string;
	category: PostCategory;
	attachmentIds?: number[];
}) {
	return api<ApiResponse<Post>>("/community/posts", {
		method: "POST",
		body: JSON.stringify(payload),
	}).then(unwrap);
}

export function updateCommunityPost(
	postId: number,
	payload: Partial<{
		title: string;
		content: string;
		category: PostCategory;
		attachmentIds: number[];
	}>,
) {
	return api<ApiResponse<Post>>(`/community/posts/${postId}`, {
		method: "PATCH",
		body: JSON.stringify(payload),
	}).then(unwrap);
}

export function deleteCommunityPost(postId: number) {
	return api<ApiResponse<{ message?: string }>>(`/community/posts/${postId}`, {
		method: "DELETE",
	}).then(unwrap);
}

export function likeCommunityPost(postId: number) {
	return api<ApiResponse<{ liked: true }>>(`/community/posts/${postId}/like`, {
		method: "POST",
	}).then(unwrap);
}

export function unlikeCommunityPost(postId: number) {
	return api<ApiResponse<{ liked: false }>>(`/community/posts/${postId}/like`, {
		method: "DELETE",
	}).then(unwrap);
}

export function createCommunityComment(postId: number, content: string) {
	return api<ApiResponse<Comment>>(`/community/posts/${postId}/comments`, {
		method: "POST",
		body: JSON.stringify({ content }),
	}).then(unwrap);
}

export function deleteCommunityComment(postId: number, commentId: number) {
	return api<ApiResponse<{ message?: string }>>(
		`/community/posts/${postId}/comments/${commentId}`,
		{ method: "DELETE" },
	).then(unwrap);
}

export async function uploadCommunityAttachments(files: File[]) {
	const fd = new FormData();
	files.forEach((f) => fd.append("files", f));

	const res = await api<ApiResponse<Attachment[]>>("/community/attachments", {
		method: "POST",
		body: fd,
	});

	return unwrap(res);
}
