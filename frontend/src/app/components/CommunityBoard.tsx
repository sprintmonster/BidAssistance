import { ChevronRight, Eye, MessageSquare, Paperclip, ThumbsUp } from "lucide-react";
import type { Post } from "./CommunityPage";

import { Badge } from "./ui/badge";
import { Card } from "./ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

interface CommunityBoardProps {
	posts: Post[];
	onSelectPost: (post: Post) => void;
}

const categoryLabels: Record<Post["category"], string> = {
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

export function CommunityBoard({ posts, onSelectPost }: CommunityBoardProps) {
	return (
		<div className="space-y-3">
			<div className="hidden md:block">
				<Card className="border bg-white overflow-hidden">
					<Table>
						<TableHeader>
							<TableRow className="bg-slate-50 hover:bg-slate-50">
								{/* ✅ '유형' 왼쪽 여백 */}
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
								const hasFile = (post.attachments?.length ?? 0) > 0;

								return (
									<TableRow
										key={post.id}
										onClick={() => onSelectPost(post)}
										className="cursor-pointer"
									>
										{/* ✅ 첫 컬럼도 동일하게 pl-6 */}
										<TableCell className="pl-6">
											<CategoryBadge category={post.category} />
										</TableCell>

										<TableCell className="min-w-0">
											<div className="flex items-center gap-2 min-w-0">
												<div className="font-medium text-gray-900 truncate">
													{post.title}
												</div>
												{hasFile && (
													<Paperclip className="h-4 w-4 text-gray-400 shrink-0" />
												)}
											</div>
											<div className="mt-0.5 text-xs text-gray-500 line-clamp-1">
												{post.content}
											</div>
										</TableCell>

										<TableCell className="text-gray-700">{post.author}</TableCell>
										<TableCell className="text-gray-500 tabular-nums">
											{post.createdAt}
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
												{post.comments.length}
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

			{/* 모바일은 기존 카드 유지 (원하면 모바일도 패딩 맞춰줄 수 있음) */}
			<div className="md:hidden space-y-3">
				{posts.map((post) => (
					<div
						key={post.id}
						onClick={() => onSelectPost(post)}
						className="bg-white rounded-lg border border-gray-200 p-4 hover:border-blue-300 hover:shadow-sm transition cursor-pointer"
					>
						<div className="flex items-center gap-2 mb-2">
							<CategoryBadge category={post.category} />
							<span className="text-xs text-gray-500">{post.author}</span>
							<span className="text-xs text-gray-400">·</span>
							<span className="text-xs text-gray-500">{post.createdAt}</span>
						</div>

						<div className="font-semibold text-gray-900 mb-1">{post.title}</div>
						<div className="text-sm text-gray-600 line-clamp-2">{post.content}</div>

						<div className="mt-3 flex items-center gap-3 text-xs text-gray-500 tabular-nums">
							<span className="inline-flex items-center gap-1">
								<Eye className="h-4 w-4" /> {post.views}
							</span>
							<span className="inline-flex items-center gap-1">
								<MessageSquare className="h-4 w-4" /> {post.comments.length}
							</span>
							<span className="inline-flex items-center gap-1">
								<ThumbsUp className="h-4 w-4" /> {post.likes}
							</span>
						</div>
					</div>
				))}

				{posts.length === 0 && (
					<div className="text-center py-12 text-gray-500">
						조건에 맞는 게시글이 없습니다.
					</div>
				)}
			</div>
		</div>
	);
}
