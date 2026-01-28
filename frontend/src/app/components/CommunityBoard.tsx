import { ChevronRight, Eye, MessageSquare, Paperclip, ThumbsUp } from "lucide-react";
import type { Post } from "../types/community";

import { Badge } from "./ui/badge";
import { Card } from "./ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

import { mask_name } from "../utils/masking";

interface CommunityBoardProps {
	posts: Post[];
	onSelectPost: (post: Post) => void;
}

const categoryLabels: Record<NonNullable<Post["category"]>, string> = {
	question: "질문",
	info: "정보",
	review: "후기",
	discussion: "토론",
};

function CategoryBadge({ category }: { category: Post["category"] }) {
	const cls =
		category === "question"
			? "border-blue-200 bg-blue-50 text-blue-700"
			: category === "info"
				? "border-emerald-200 bg-emerald-50 text-emerald-700"
				: category === "review"
					? "border-violet-200 bg-violet-50 text-violet-700"
					: "border-amber-200 bg-amber-50 text-amber-700";

	return (
		<Badge variant="outline" className={cls}>
			{categoryLabels[category]}
		</Badge>
	);
}
function formatCreatedAt(input: unknown) {
    if (!input) return "";

    // 1) Date 객체면 그대로
    if (input instanceof Date) {
        return new Intl.DateTimeFormat("ko-KR", {
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
        }).format(input);
    }

    // 2) 문자열/숫자면 Date로 파싱 시도
    const s = String(input).trim();

    // "2026-01-28 15:03:00" 같은 형태도 파싱되게 공백을 'T'로 바꿔줌
    const normalized = s.includes(" ") && !s.includes("T") ? s.replace(" ", "T") : s;

    const d = new Date(normalized);
    if (Number.isNaN(d.getTime())) return s; // 파싱 실패하면 원문 유지

    return new Intl.DateTimeFormat("ko-KR", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
    }).format(d);
}

export function CommunityBoard({ posts, onSelectPost }: CommunityBoardProps) {
	return (
		<div className="space-y-3">
			<div className="hidden md:block">
				<Card className="border bg-white overflow-hidden">
					<Table>
						<TableHeader>
							<TableRow className="bg-slate-50 hover:bg-slate-50">
								<TableHead className="w-[90px] pl-6">유형</TableHead>
								<TableHead>제목</TableHead>
								<TableHead className="w-[140px]">작성자</TableHead>
								<TableHead className="w-[120px]">작성일</TableHead>
								<TableHead className="w-[90px] text-right">조회</TableHead>
								<TableHead className="w-[90px] text-right">댓글</TableHead>
								<TableHead className="w-[90px] text-right">좋아요</TableHead>
								<TableHead className="w-[44px]" />
							</TableRow>
						</TableHeader>

						<TableBody>
							{posts.map((post) => {
								const commentCount = post.commentCount ?? (post.comments?.length ?? 0);
								const hasFile = (post.attachmentCount ?? (post.attachments?.length ?? 0)) > 0;

								return (
									<TableRow
										key={post.id}
										onClick={() => onSelectPost(post)}
										className="cursor-pointer"
									>
										<TableCell className="pl-6">
											<CategoryBadge category={post.category} />
										</TableCell>

										<TableCell className="min-w-0">
											<div className="flex items-center gap-2 min-w-0">
												<div className="font-medium text-gray-900 truncate">
													{post.title}
												</div>
												{hasFile && <Paperclip className="h-4 w-4 text-gray-400 shrink-0" />}
											</div>
											<div className="mt-0.5 text-xs text-gray-500 line-clamp-1">
												{post.contentPreview ?? post.content ?? ""}
											</div>
										</TableCell>

									<TableCell className="text-gray-700">{mask_name(post.authorName)}</TableCell>
										<TableCell className="text-gray-500 tabular-nums">
                                            {formatCreatedAt(post.createdAt)}
                                        </TableCell>

										<TableCell className="text-right text-gray-600 tabular-nums">
											<span className="inline-flex items-center gap-1 justify-end">
												<Eye className="h-4 w-4 text-gray-400" />
												{post.views}
											</span>
										</TableCell>

										<TableCell className="text-right text-gray-600 tabular-nums">
											<span className="inline-flex items-center gap-1 justify-end">
												<MessageSquare className="h-4 w-4 text-gray-400" />
												{commentCount}
											</span>
										</TableCell>

										<TableCell className="text-right text-gray-600 tabular-nums">
											<span className="inline-flex items-center gap-1 justify-end">
												<ThumbsUp className="h-4 w-4 text-gray-400" />
												{post.likes}
											</span>
										</TableCell>

										<TableCell className="text-right">
											<ChevronRight className="h-4 w-4 text-gray-400" />
										</TableCell>
									</TableRow>
								);
							})}

							{posts.length === 0 && (
								<TableRow>
									<TableCell colSpan={8} className="py-12 text-center text-gray-500">
										조건에 맞는 게시글이 없습니다.
									</TableCell>
								</TableRow>
							)}
						</TableBody>
					</Table>
				</Card>
			</div>

			{/* 모바일 카드 */}
			<div className="md:hidden space-y-3">
				{posts.map((post) => {
					const commentCount = post.commentCount ?? (post.comments?.length ?? 0);
					return (
						<div
							key={post.id}
							onClick={() => onSelectPost(post)}
							className="bg-white rounded-lg border border-gray-200 p-4 hover:border-blue-300 hover:shadow-sm transition cursor-pointer"
						>
							<div className="flex items-center gap-2 mb-2">
								<CategoryBadge category={post.category} />
									<span className="text-xs text-gray-500">{mask_name(post.authorName)}</span>
								<span className="text-xs text-gray-400">·</span>
								<span className="text-xs text-gray-500">{formatCreatedAt(post.createdAt)}</span>
							</div>

							<div className="font-semibold text-gray-900 mb-1">{post.title}</div>
							<div className="text-sm text-gray-600 line-clamp-2">
								{post.contentPreview ?? post.content ?? ""}
							</div>

							<div className="mt-3 flex items-center gap-3 text-xs text-gray-500 tabular-nums">
								<span className="inline-flex items-center gap-1">
									<Eye className="h-4 w-4" /> {post.views}
								</span>
								<span className="inline-flex items-center gap-1">
									<MessageSquare className="h-4 w-4" /> {commentCount}
								</span>
								<span className="inline-flex items-center gap-1">
									<ThumbsUp className="h-4 w-4" /> {post.likes}
								</span>
							</div>
						</div>
					);
				})}

				{posts.length === 0 && (
					<div className="text-center py-12 text-gray-500">
						조건에 맞는 게시글이 없습니다.
					</div>
				)}
			</div>
		</div>
	);
}
