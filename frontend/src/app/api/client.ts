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

const DOMAIN = import.meta.env.VITE_API_URL || "";
const BASE_URL = `${DOMAIN}/api`;

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
		`/board/posts${qs({
			category: opts.category,
			q: opts.q,
			sort: opts.sort,
			page: opts.page,
			size: opts.size,
		})}`,
	).then(unwrap);
}

export function fetchCommunityPost(postId: number) {
	return api<ApiResponse<Post>>(`/board/posts/${postId}`).then(unwrap);
}

export function createCommunityPost(payload: {
	title: string;
	content: string;
	category: PostCategory;
	attachmentIds?: string[];
}) {
	return api<ApiResponse<Post>>("/board/posts", {
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
		attachmentIds: string[];
	}>,
) {
	return api<ApiResponse<Post>>(`/board/posts/${postId}`, {
		method: "PATCH",
		body: JSON.stringify(payload),
	}).then(unwrap);
}

export function deleteCommunityPost(postId: number) {
	return api<ApiResponse<{ message?: string }>>(`/board/posts/${postId}`, {
		method: "DELETE",
	}).then(unwrap);
}

export function likeCommunityPost(postId: number) {
	return api<ApiResponse<{ liked: true }>>(`/board/posts/${postId}/like`, {
		method: "POST",
	}).then(unwrap);
}

export function unlikeCommunityPost(postId: number) {
	return api<ApiResponse<{ liked: false }>>(`/board/posts/${postId}/dislike`, {
		method: "DELETE",
	}).then(unwrap);
}

export function createCommunityComment(postId: number, content: string) {
	return api<ApiResponse<Comment>>(`/board/posts/${postId}/comments`, {
		method: "POST",
		body: JSON.stringify({ content }),
	}).then(unwrap);
}

export function deleteCommunityComment(postId: number, commentId: string) {
	return api<ApiResponse<{ message?: string }>>(`/comments/${commentId}`, {
		method: "DELETE",
	}).then(unwrap);
}

export async function uploadCommunityAttachments(files: File[]) {
	const fd = new FormData();
	files.forEach((f) => fd.append("files", f));

	// userId 기반 접근통제: 백엔드가 읽는 헤더명으로 통일 필요(여기서는 X-User-Id 사용)
	const userId = localStorage.getItem("userId");
	const headers: Record<string, string> = {};
	if (userId) headers["X-User-Id"] = String(userId);

	// 배포 환경에서도 동작하도록 BASE_URL 사용 + 쿠키 기반 세션 대비 credentials 포함
	const res = await fetch(`${BASE_URL}/board/attachments`, {
		method: "POST",
		credentials: "include",
		headers,
		body: fd,
	});

	if (!res.ok) {
		const msg = await res.text().catch(() => "");
		throw new Error(msg || "첨부 업로드 실패");
	}

	const json = (await res.json()) as ApiResponse<Attachment[]>;
	return unwrap(json);
}
