export type Id = string | number;
export type PostId = number;
export type CommentId = number;
export type AttachmentId = number;
export type PostCategory = "question" | "info" | "review" | "discussion";

export interface Attachment {
	id: AttachmentId;            // <- 변경
	name: string;
	type: string;
	url: string;
	size: number;
	isImage: boolean;
}

export interface Comment {
	id:CommentId;            // <- 변경
	authorId?: string;
	authorName: string;
	content: string;
	createdAt: string;
	likes: number;
	likedByMe?: boolean;
}

export interface Post {
	id: PostId;            // <- 변경
	title: string;

	content?: string;
	contentPreview?: string;

	category: PostCategory;

	authorId?: string;
	authorName: string;

	createdAt: string;
	views: number;
	likes: number;
	likedByMe: boolean;

	commentCount?: number;
	comments?: Comment[];

	attachmentCount?: number;
	attachments?: Attachment[];
}

export type SortKey = "latest" | "popular" | "views" | "comments";

export interface PostListData {
	items: Post[];
	page: number;
	size: number;
	total: number;
	counts?: Record<"all" | PostCategory, number>;
}

export interface ApiResponse<T> {
	status: "success" | "error";
	message?: string;
	data: T;
}
