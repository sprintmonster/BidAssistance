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

function normalizePostFromDetail(raw: any): Post {
    const id = toNum(raw?.id ?? raw?.postId);
    if (!Number.isFinite(id) || id <= 0) throw new Error(`게시글 ID가 올바르지 않습니다. (id=${String(raw?.id)})`);

    const attachments = (raw?.attachments ?? []).map(normalizeAttachment);

    return {
        id,
        title: toStr(raw?.title),
        content: toStr(raw?.content ?? ""),
        contentPreview: toStr(raw?.contentPreview ?? ""),
        category: (raw?.category ?? "question"),

        authorId: raw?.authorId != null ? String(raw.authorId) : undefined,
        authorName: toStr(raw?.userName ?? raw?.authorName ?? "—"),

        createdAt: toStr(raw?.createdAt),
        views: toNum(raw?.viewCount ?? raw?.views ?? 0),
        likes: toNum(raw?.likeCount ?? raw?.likes ?? 0),
        likedByMe: !!raw?.likedByMe,

        commentCount: toNum(raw?.commentCount ?? 0),
        comments: raw?.comments ?? [],

        attachmentCount: toNum(raw?.attachmentCount ?? attachments.length),
        attachments,
    };
}

function normalizePostFromList(it: any): Post {
    const id = toNum(it?.postId ?? it?.id);
    if (!Number.isFinite(id) || id <= 0) throw new Error(`목록 게시글 ID 오류 (id=${String(it?.postId ?? it?.id)})`);

    return {
        id,
        title: toStr(it?.title),
        contentPreview: toStr(it?.contentPreview ?? ""),
        category: (it?.category ?? "question"),

        authorId: it?.authorId != null ? String(it.authorId) : undefined,
        authorName: toStr(it?.authorName ?? "—"),

        createdAt: toStr(it?.createdAt),
        views: toNum(it?.views ?? 0),
        likes: toNum(it?.likes ?? 0),
        likedByMe: !!it?.likedByMe,

        commentCount: toNum(it?.commentCount ?? 0),
        comments: it?.comments ?? [],

        attachmentCount: toNum(it?.attachmentCount ?? 0),
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
    return api<ApiResponse<PostListData>>(`/board/posts${qs({
        category: opts.category,
        q: opts.q,
        sort: opts.sort,
        page: opts.page,
        size: opts.size,
    })}`)
        .then(unwrap)
        .then((data: any) => {
            const items = (data?.items ?? []).map(normalizePostFromList);
            return { ...data, items };
        });
}


// export function fetchCommunityPost(postId: number) {
//     return api<ApiResponse<Post>>(`/board/${postId}`)
//         .then(unwrap)
//         .then((it: any) => ({
//             ...it,
//             id: it.postId ?? it.id,
//             authorId: it.authorId != null ? String(it.authorId) : undefined,
//             comments: it.comments ?? [],
//             attachments: it.attachments ?? [],
//         }));
// }
export function fetchCommunityPost(postId: number) {
    return api<ApiResponse<any>>(`/board/${postId}`)
        .then(unwrap)
        .then((raw) => normalizePostFromDetail(raw));
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

export function updateCommunityPost(postId: number, payload: Partial<{
    title: string;
    content: string;
    category: PostCategory;
    attachmentIds: string[];
}>) {
    return api<ApiResponse<any>>(`/board/${postId}`, {
        method: "POST", // 백엔드: @PostMapping("/{id}")
        body: JSON.stringify(payload),
    })
        .then(unwrap)
        .then((raw) => normalizePostFromDetail(raw));
}

export function deleteCommunityPost(postId: number) {
    return api<ApiResponse<{ message?: string }>>(`/board/${postId}`, {
        method: "DELETE",
    }).then(unwrap);
}

export function likeCommunityPost(postId: number) {
    return api<ApiResponse<any>>(`/board/posts/${postId}/like`, {
        method: "POST",
    }).then(unwrap);
}

export function unlikeCommunityPost(postId: number) {
    return api<ApiResponse<any>>(`/board/posts/${postId}/dislike`, {
        method: "POST",
    }).then(unwrap);
}


export function createCommunityComment(postId: number, content: string) {
    return api<ApiResponse<Comment>>(`/boards/${postId}/comments`, {
        method: "POST",
        body: JSON.stringify({ content }),
    }).then(unwrap);
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




function toNumId(v: any, fallback?: number): number {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
    if (fallback != null && Number.isFinite(Number(fallback))) return Number(fallback);
    throw new Error(`게시글 ID가 올바르지 않습니다. (id=${String(v)})`);
}

function normalizePost(raw: any, fallbackId?: number) {
    const id = toNumId(raw?.postId ?? raw?.id ?? raw?.boardId, fallbackId);

    return {
        ...raw,
        id,
        authorId: raw?.authorId != null ? String(raw.authorId) : undefined,
        comments: raw?.comments ?? [],
        attachments: raw?.attachments ?? [],
        views: Number(raw?.views ?? 0),
        likes: Number(raw?.likes ?? 0),
        likedByMe: !!raw?.likedByMe,
        commentCount: raw?.commentCount != null ? Number(raw.commentCount) : (raw?.comments?.length ?? 0),
        attachmentCount: raw?.attachmentCount != null ? Number(raw.attachmentCount) : (raw?.attachments?.length ?? 0),
    };
}


