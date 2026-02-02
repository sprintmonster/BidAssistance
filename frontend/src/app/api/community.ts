import { api } from "./client";
import type { ApiResponse, Post, PostListData, SortKey, PostCategory, Comment, Attachment } from "../types/community";

function toNum(v: any, fallback = 0) {
	const n = Number(v);
	return Number.isFinite(n) ? n : fallback;
}

function toStr(v: any, fallback = "") {
	return typeof v === "string" ? v : v == null ? fallback : String(v);
}

function isImageByName(name: string) {
	return /\.(png|jpe?g|gif|webp|bmp|svg)$/i.test(name);
}

function pickAuthorId(raw: any): string | undefined {
	const v =
		raw?.authorId ??
		raw?.userId ??
		raw?.writerId ??
		raw?.memberId ??
		raw?.user?.id ??
		raw?.user?.userId ??
		raw?.member?.id;

	if (v == null) return undefined;
	const s = String(v);
	return s ? s : undefined;
}

function normalizeAttachment(a: any): Attachment {
	const name = toStr(a?.name ?? a?.fileName ?? "file");
	const url = toStr(a?.url ?? "");
	return {
		id: toNum(a?.id),
		name,
		type: toStr(a?.type ?? ""),
		url,
		size: toNum(a?.size ?? 0),
		isImage: !!a?.isImage || isImageByName(name) || url.startsWith("data:image"),
	};
}

function normalizeComment(raw: any): Comment {
	const id = toNum(raw?.id ?? raw?.commentId);
	if (!Number.isFinite(id) || id <= 0) {
		throw new Error(`댓글 ID가 올바르지 않습니다. (id=${String(raw?.id ?? raw?.commentId)})`);
	}

	return {
		id,
		authorId: pickAuthorId(raw),
		authorName: toStr(raw?.authorName ?? raw?.userName ?? raw?.author ?? "—"),
		content: toStr(raw?.content ?? ""),
		createdAt: toStr(raw?.createdAt ?? ""),
		likes: toNum(raw?.likes ?? raw?.likeCount ?? 0),
		likedByMe: !!raw?.likedByMe,
	};
}

function normalizePostFromDetail(raw: any): Post {
	const id = toNum(raw?.id ?? raw?.postId);
	if (!Number.isFinite(id) || id <= 0) throw new Error(`게시글 ID가 올바르지 않습니다. (id=${String(raw?.id)})`);

	const attachments = (raw?.attachments ?? []).map(normalizeAttachment);

	return {
		id,
		title: toStr(raw?.title),
		content: toStr(raw?.content ?? ""),
		contentPreview: toStr(raw?.contentPreview ?? ""),
		category: raw?.category ?? "question",
		authorId: raw?.authorId != null ? String(raw.authorId) : undefined,
		authorName: toStr(raw?.userName ?? raw?.authorName ?? "—"),
		createdAt: toStr(raw?.createdAt),
		views: toNum(raw?.viewCount ?? raw?.views ?? 0),
		likes: toNum(raw?.likeCount ?? raw?.likes ?? 0),
		likedByMe: !!raw?.likedByMe,
		comments: (raw?.comments ?? []).map(normalizeComment),
		commentCount: toNum(raw?.commentCount ?? (raw?.comments?.length ?? 0)),
		attachmentCount: toNum(raw?.attachmentCount ?? attachments.length),
		attachments,
	};
}

function normalizePostFromList(it: any): Post {
	const id = toNum(it?.postId ?? it?.id);
	if (!Number.isFinite(id) || id <= 0) throw new Error(`목록 게시글 ID 오류 (id=${String(it?.postId ?? it?.id)})`);

	const listAttachments =
		it?.attachments ??
		it?.files ??
		it?.fileList ??
		it?.attachFiles ??
		it?.attachmentList ??
		[];

	const rawCount =
		it?.attachmentCount ??
		it?.attachmentsCount ??
		it?.attachmentCnt ??
		it?.fileCount ??
		it?.filesCount ??
		it?.fileCnt ??
		it?.attachCount ??
		it?.attachCnt;

	let attachmentCount = toNum(rawCount, -1);
	if (attachmentCount < 0) {
		if (it?.hasAttachment === true || it?.hasAttachments === true || it?.hasFile === true)
			attachmentCount = 1;
		else if (Array.isArray(listAttachments) && listAttachments.length > 0)
			attachmentCount = listAttachments.length;
		else attachmentCount = 0;
	}

	return {
		id,
		title: toStr(it?.title),
		contentPreview: toStr(it?.contentPreview ?? ""),
		category: it?.category ?? "question",
		authorId: it?.authorId != null ? String(it.authorId) : undefined,
		authorName: toStr(it?.authorName ?? "—"),
		createdAt: toStr(it?.createdAt),
		views: toNum(it?.views ?? 0),
		likes: toNum(it?.likes ?? 0),
		likedByMe: !!it?.likedByMe,
		commentCount: toNum(it?.commentCount ?? 0),
		comments: it?.comments ?? [],
		attachmentCount,
		attachments: it?.attachments ?? [],
		content: it?.content,
	};
}

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
	)
		.then(unwrap)
		.then((data: any) => {
			const items = (data?.items ?? []).map(normalizePostFromList);
			return { ...data, items };
		});
}

export function fetchCommunityPost(postId: number) {
	return api<ApiResponse<any>>(`/board/${postId}`).then(unwrap).then((raw) => normalizePostFromDetail(raw));
}

export function createCommunityPost(payload: {
	title: string;
	content: string;
	category: PostCategory;
	attachmentIds?: string[];
}) {
	return api<ApiResponse<any>>("/board", {
		method: "POST",
		body: JSON.stringify(payload),
	})
		.then(unwrap)
		.then((raw) => normalizePostFromDetail(raw));
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
	return api<ApiResponse<any>>(`/board/${postId}`, {
		method: "POST",
		body: JSON.stringify(payload),
	})
		.then(unwrap)
		.then((raw) => normalizePostFromDetail(raw));
}

export function deleteCommunityPost(postId: number) {
	return api<ApiResponse<{ message?: string }>>(`/board/${postId}`, { method: "DELETE" }).then(unwrap);
}

export function likeCommunityPost(postId: number) {
	return api<ApiResponse<null>>(`/board/posts/${postId}/like`, { method: "POST" }).then(unwrap);
}

export function unlikeCommunityPost(postId: number) {
	return api<ApiResponse<null>>(`/board/posts/${postId}/dislike`, { method: "POST" }).then(unwrap);
}

export function fetchCommunityComments(boardId: number) {
	return api<ApiResponse<any[]>>(`/boards/${boardId}/comments`, { method: "GET" })
		.then(unwrap)
		.then((list) => (list ?? []).map(normalizeComment));
}

export function createCommunityComment(postId: number, content: string) {
	const userIdRaw = localStorage.getItem("userId");
	const userId = userIdRaw ? Number(userIdRaw) : NaN;
	if (!Number.isFinite(userId) || userId <= 0) {
		throw new Error("로그인이 필요합니다. (userId 없음)");
	}

	return api<ApiResponse<any>>(`/boards/${postId}/comments`, {
		method: "POST",
		body: JSON.stringify({ content, userId }),
	})
		.then(unwrap)
		.then(normalizeComment);
}

export function deleteCommunityComment(commentId: string) {
	const userId = localStorage.getItem("userId");
	if (!userId) throw new Error("로그인이 필요합니다.");

	return api<ApiResponse<{ message?: string }>>(`/comments/${commentId}?userId=${userId}`, {
		method: "DELETE",
	}).then(unwrap);
}

export async function uploadCommunityAttachments(files: File[]) {
	const fd = new FormData();
	files.forEach((f) => fd.append("files", f));

	const json = await api<ApiResponse<any[]>>("/board/attachments", {
		method: "POST",
		body: fd,
	});

	const rawList = unwrap(json) ?? [];
	return rawList.map(normalizeAttachment);
}
