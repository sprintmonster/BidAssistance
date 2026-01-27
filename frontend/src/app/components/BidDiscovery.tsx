import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Eye, FilterX, Plus, RefreshCw, Search } from "lucide-react";
import { fetchWishlist, toggleWishlist } from "../api/wishlist";
import { fetchBids } from "../api/bids";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "./ui/card";
// import {
// 	Dialog,
// 	DialogContent,
// 	DialogDescription,
// 	DialogHeader,
// 	DialogTitle,
// } from "./ui/dialog";
import { Input } from "./ui/input";
import {
	Pagination,
	PaginationContent,
	PaginationItem,
	PaginationLink,
	PaginationNext,
	PaginationPrevious,
} from "./ui/pagination";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "./ui/select";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "./ui/table";
import { cn } from "./ui/utils";

type SortKey = "deadline_asc" | "deadline_desc" | "title_asc";

type UiBid = {
	bidId: number;
	realId: string;
	title: string;
	agency: string;
	budget: string;
	deadline: string;
};

function parseDate(value: string) {
	const trimmed = (value || "").trim();
	if (!trimmed) return null;
	const d = new Date(trimmed);
	if (!Number.isFinite(d.getTime())) return null;
	return d;
}

function diffDays(from: Date, to: Date) {
	const ms = to.getTime() - from.getTime();
	return Math.ceil(ms / (1000 * 60 * 60 * 24));
}

function formatDday(deadline: string) {
	const d = parseDate(deadline);
	if (!d) return null;
	const days = diffDays(new Date(), d);
	if (days === 0) return "D-DAY";
	if (days > 0) return `D-${days}`;
	return `D+${Math.abs(days)}`;
}

function isEnded(deadline: string) {
	const d = parseDate(deadline);
	if (!d) return false;
	return d.getTime() < Date.now();
}

export function BidDiscovery({
	setGlobalLoading,
	showToast,
}: {
	setGlobalLoading: (v: boolean) => void;
	showToast: (msg: string, type: "success" | "error") => void;
}) {
	const location = useLocation();
	const urlQuery = useMemo(() => {
		const q = new URLSearchParams(location.search).get("q");
		return (q || "").trim();
	}, [location.search]);
	const [wishlistSynced, setWishlistSynced] = useState(false);

	const [bids, setBids] = useState<UiBid[]>([]);
	const [keyword, setKeyword] = useState<string>(urlQuery);
	const [agency, setAgency] = useState<string>("all");
	const [sortKey, setSortKey] = useState<SortKey>("deadline_asc");
	const [page, setPage] = useState<number>(1);
	const [pageSize, setPageSize] = useState<number>(10);
	// const [selected, setSelected] = useState<UiBid | null>(null);

	const [addingId, setAddingId] = useState<number | null>(null);
	const [addedIds, setAddedIds] = useState<Set<number>>(() => new Set());

	const navigate = useNavigate();

	useEffect(() => {
		setKeyword(urlQuery);
		setPage(1);
	}, [urlQuery]);

	const load = async () => {
		console.log("load() start");
		try {
			setGlobalLoading(true);
			const res = await fetchBids();
			console.log("after fetchBids()");
			console.log("fetchBids res:", res);
			const items = Array.isArray(res)
				? res
				: Array.isArray((res as any)?.data)
					? (res as any).data
					: Array.isArray((res as any)?.data?.items)
						? (res as any).data.items
						: [];

			const mapped: UiBid[] = items
				.map((it: any) => {
					const bidId = Number(it.bidId ?? it.id); // ✅ int
					const realId = String(it.realId ?? it.bidNo ?? "");
					if (!Number.isFinite(bidId)) return null;

					return {
						bidId,
						realId,
						title: String(it.title ?? it.name ?? ""),
						agency: String(it.agency ?? it.organization ?? ""),
						budget:
							it.baseAmount != null ? String(it.baseAmount) :
							it.estimatePrice != null ? String(it.estimatePrice) :
							"",
						deadline: String(it.bidEnd ?? it.endDate ?? ""),
					} as UiBid;
				})
				.filter(Boolean) as UiBid[];

			setBids(mapped);
		} catch (e) {
			console.error("load() failed:", e);
			showToast("공고 목록을 불러오지 못했습니다.", "error");
			setBids([]);
		} finally {
			setGlobalLoading(false);
		}
	};

	useEffect(() => {
		void load();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	useEffect(() => {
		const syncAddedFromServer = async () => {
			setWishlistSynced(false);

			const userIdStr = localStorage.getItem("userId");
			const userId = Number(userIdStr);

			if (!userIdStr || !Number.isFinite(userId)) {
				setAddedIds(new Set());
				setWishlistSynced(true);
				return;
			}

			try {
				const items = await fetchWishlist(userId);
				setAddedIds(new Set(items.map((it) => it.bidId)));
			} catch {
				setAddedIds(new Set());
			} finally {
				setWishlistSynced(true);
			}
		};

		void syncAddedFromServer();
	}, []);

	const agencies = useMemo(() => {
		const set = new Set<string>();
		bids.forEach((b) => {
			const a = (b.agency || "").trim();
			if (a) set.add(a);
		});
		return ["all", ...Array.from(set).sort((a, b) => a.localeCompare(b))];
	}, [bids]);

	const filtered = useMemo(() => {
		const q = keyword.trim().toLowerCase();
		let list = bids.slice();

		list = list.filter((b) => !isEnded(b.deadline));

		if (agency !== "all") {
			list = list.filter((b) => (b.agency || "").trim() === agency);
		}

		if (q) {
			list = list.filter((b) => {
				const hay = `${b.title} ${b.agency} ${b.budget} ${b.deadline} ${b.realId}`.toLowerCase();
				return hay.includes(q);
			});
		}

		list.sort((a, b) => {
			if (sortKey === "title_asc") return a.title.localeCompare(b.title);

			const ad = parseDate(a.deadline)?.getTime() ?? Number.POSITIVE_INFINITY;
			const bd = parseDate(b.deadline)?.getTime() ?? Number.POSITIVE_INFINITY;
			return sortKey === "deadline_desc" ? bd - ad : ad - bd;
		});

		return list;
	}, [bids, agency, keyword, sortKey]);

	const total = filtered.length;
	const totalPages = Math.max(1, Math.ceil(total / pageSize));
	const safePage = Math.min(Math.max(1, page), totalPages);

	const paged = useMemo(() => {
		const start = (safePage - 1) * pageSize;
		return filtered.slice(start, start + pageSize);
	}, [filtered, safePage, pageSize]);

	useEffect(() => {
		setPage(safePage);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [totalPages]);

	const resetFilters = () => {
		setKeyword("");
		setAgency("all");
		setSortKey("deadline_asc");
		setPage(1);
	};

	const addToCart = async (bidId: number) => {
		try {
			const userIdStr = localStorage.getItem("userId");
			const userId = Number(userIdStr);

			if (!userIdStr || !Number.isFinite(userId)) {
				showToast("로그인이 필요합니다. 다시 로그인 해주세요.", "error");
				return;
			}

			setAddingId(bidId);
			setGlobalLoading(true);

			const res = await toggleWishlist(userId, bidId);

			if (res.status !== "success") {
				showToast(res.message || "추가 실패", "error");
				return;
			}

			setAddedIds((prev) => {
				const next = new Set(prev);
				next.add(bidId);
				return next;
			});

			const items = await fetchWishlist(userId);
			setAddedIds(new Set(items.map((it) => it.bidId)));

			showToast("장바구니에 추가됨", "success");
		} catch (e) {
			showToast("추가 실패", "error");
		} finally {
			setGlobalLoading(false);
			setAddingId(null);
		}
	};

	const paginationNumbers = useMemo(() => {
		const numbers: number[] = [];
		const windowSize = 5;
		const half = Math.floor(windowSize / 2);
		let start = Math.max(1, safePage - half);
		let end = Math.min(totalPages, start + windowSize - 1);
		start = Math.max(1, end - windowSize + 1);
		for (let i = start; i <= end; i += 1) numbers.push(i);
		return numbers;
	}, [safePage, totalPages]);

	function formatDateTimeLines(dateStr: string) {
		if (!dateStr) return { dateLine: "-", timeLine: "" };

		const d = new Date(dateStr);
		if (!Number.isFinite(d.getTime())) return { dateLine: "-", timeLine: "" };

		const dateLine = d.toLocaleDateString("ko-KR", {
			year: "numeric",
			month: "2-digit",
			day: "2-digit",
		});

		const timeLine = d.toLocaleTimeString("ko-KR", {
			hour: "2-digit",
			minute: "2-digit",
			hour12: true, // "오전/오후" 나오게
		});

		return { dateLine, timeLine };
	}

	return (
		<div className="space-y-4">
			<Card>
				<CardHeader className="space-y-1">
					<CardTitle className="text-xl">공고 찾기</CardTitle>
					<CardDescription>
						키워드/기관/정렬 기반으로 공고를 빠르게 찾고, 장바구니에 담아 관리하세요.
					</CardDescription>
				</CardHeader>

				<CardContent className="space-y-3">
					<div className="flex flex-col gap-2 lg:flex-row lg:items-center">
						<div className="flex flex-1 items-center gap-2">
							<div className="relative flex-1">
								<Search className="pointer-events-none absolute left-2 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
								<Input
									value={keyword}
									onChange={(e) => {
										setKeyword(e.target.value);
										setPage(1);
									}}
									placeholder={urlQuery ? `검색어: ${urlQuery}` : "키워드 검색 (공고명/기관/예산/마감)"}
									className="pl-8"
								/>
							</div>

							<Button variant="outline" onClick={() => void load()} className="shrink-0">
								<RefreshCw className="mr-2 size-4" />
								새로고침
							</Button>

							<Button variant="ghost" onClick={resetFilters} className="shrink-0">
								<FilterX className="mr-2 size-4" />
								필터 초기화
							</Button>
						</div>

						<div className="flex flex-col gap-2 sm:flex-row sm:items-center">
							<Select
								value={agency}
								onValueChange={(v) => {
									setAgency(v);
									setPage(1);
								}}
							>
								<SelectTrigger className="w-[220px]">
									<SelectValue placeholder="기관" />
								</SelectTrigger>
								<SelectContent>
									{agencies.map((a) => (
										<SelectItem key={a} value={a}>
											{a === "all" ? "전체 기관" : a}
										</SelectItem>
									))}
								</SelectContent>
							</Select>

							<Select value={sortKey} onValueChange={(v) => setSortKey(v as SortKey)}>
								<SelectTrigger className="w-[200px]">
									<SelectValue placeholder="정렬" />
								</SelectTrigger>
								<SelectContent>
									<SelectItem value="deadline_asc">마감 빠른 순</SelectItem>
									<SelectItem value="deadline_desc">마감 늦은 순</SelectItem>
									<SelectItem value="title_asc">공고명 A→Z</SelectItem>
								</SelectContent>
							</Select>

							<div className="flex items-center gap-2 sm:ml-auto">
								<Select
									value={String(pageSize)}
									onValueChange={(v) => {
										setPageSize(Number(v));
										setPage(1);
									}}
								>
									<SelectTrigger className="w-[160px]">
										<SelectValue placeholder="페이지 크기" />
									</SelectTrigger>
									<SelectContent>
										<SelectItem value="10">10개씩</SelectItem>
										<SelectItem value="20">20개씩</SelectItem>
										<SelectItem value="50">50개씩</SelectItem>
									</SelectContent>
								</Select>

								<div className="text-sm text-muted-foreground">
									총 <span className="font-medium text-foreground">{total}</span>건
								</div>
							</div>
						</div>
					</div>

					{/* ✅ Table wrapper에 배경/그림자 추가 + 헤더 배경 */}
					<div className="rounded-lg border bg-white shadow-sm">
						<Table>
							<TableHeader className="bg-slate-50/60">
								<TableRow className="hover:bg-transparent">
									<TableHead className="w-[120px] pl-6">마감</TableHead>
									<TableHead>공고명</TableHead>
									<TableHead className="w-[220px]">발주기관</TableHead>

									{/* ✅ 숫자 컬럼: 오른쪽 정렬 + 우측 패딩 */}
									<TableHead className="w-[160px] pr-6 text-right">예산</TableHead>

									{/* ✅ 상태: 중앙 정렬 */}
									<TableHead className="w-[140px] text-center">상태</TableHead>

									{/* ✅ 액션: 오른쪽 정렬 + 우측 패딩 */}
									<TableHead className="w-[180px] pr-6 text-right">액션</TableHead>
								</TableRow>
							</TableHeader>

							<TableBody>
								{paged.length === 0 ? (
									<TableRow>
										<TableCell colSpan={6} className="py-10 text-center text-muted-foreground">
											조건에 맞는 공고가 없습니다.
										</TableCell>
									</TableRow>
								) : (
									paged.map((b) => {
										const dday = formatDday(b.deadline);
										const alreadyAdded = addedIds.has(b.bidId);

										const statusVariant = dday === "D-DAY" ? "destructive" : "secondary";
										const statusLabel = dday === "D-DAY" ? "오늘 마감" : dday ? "진행중" : "확인 필요";

										return (
											<TableRow
												key={`${b.bidId}-${b.realId}`}
												className="cursor-pointer"
												onClick={() => navigate(`/bids/${b.bidId}`)}
											>
												<TableCell className="whitespace-normal pl-6">
													<div className="flex flex-col">
														{(() => {
															const { dateLine, timeLine } = formatDateTimeLines(b.deadline);

															return (
																<div className="flex flex-col">
																	<span className="text-sm font-medium">
																		{dday || dateLine}
																	</span>

																	{dday ? (
																		<span className="text-xs text-muted-foreground">
																			{dateLine} <br /> {timeLine}
																		</span>
																	) : (
																		timeLine && (
																			<span className="text-xs text-muted-foreground">
																				{timeLine}
																			</span>
																		)
																	)}
																</div>
															);
														})()}
													</div>
												</TableCell>

												<TableCell className="whitespace-normal">
													<div className="line-clamp-2 font-medium">{b.title}</div>
													<div className="text-xs text-muted-foreground">{b.realId}</div>
												</TableCell>

												<TableCell className="whitespace-normal">
													<div className="line-clamp-2">{b.agency}</div>
												</TableCell>

												{/* ✅ 숫자 가독성: tabular-nums + 우측 패딩 */}
												<TableCell className="pr-6 text-right tabular-nums">
													{Number(b.budget).toLocaleString()}
												</TableCell>

												{/* ✅ 상태: 중앙 정렬 + 뱃지 padding 통일 */}
												<TableCell className="text-center">
													<Badge variant={statusVariant} className="px-2.5">
														{statusLabel}
													</Badge>
												</TableCell>

												{/* ✅ 액션: 우측 패딩 + 버튼 높이/라운드 통일 */}
												<TableCell className="pr-6">
													<div className="flex justify-end gap-2">
														<Button
															variant="outline"
															size="sm"
															className="h-9 rounded-xl"
															onClick={(e) => {
																e.stopPropagation();
																navigate(`/bids/${b.bidId}`);
															}}
														>
															<Eye className="mr-2 size-4" />
															상세
														</Button>

														<Button
															size="sm"
															disabled={addingId === b.bidId || !wishlistSynced}
															className={cn("h-9 rounded-xl", alreadyAdded && "opacity-70")}
															onClick={(e) => {
																e.stopPropagation();

																if (!wishlistSynced) {
																	showToast("장바구니 상태를 불러오는 중입니다. 잠시 후 다시 시도해 주세요.", "error");
																	return;
																}

																if (alreadyAdded) {
																	showToast("이미 장바구니에 담긴 공고입니다.", "success");
																	return;
																}

																void addToCart(b.bidId);
															}}
														>
															<Plus className="mr-2 size-4" />
															{alreadyAdded ? "담김" : "담기"}
														</Button>
													</div>
												</TableCell>
											</TableRow>
										);
									})
								)}
							</TableBody>
						</Table>
					</div>

					{totalPages > 1 && (
						<div className="border-t py-3">
							<Pagination>
								<PaginationContent>
									<PaginationItem>
										<PaginationPrevious
											href="#"
											onClick={(e) => {
												e.preventDefault();
												setPage((p) => Math.max(1, p - 1));
											}}
										/>
									</PaginationItem>

									{paginationNumbers.map((n) => (
										<PaginationItem key={n}>
											<PaginationLink
												href="#"
												isActive={n === safePage}
												onClick={(e) => {
													e.preventDefault();
													setPage(n);
												}}
											>
												{n}
											</PaginationLink>
										</PaginationItem>
									))}

									<PaginationItem>
										<PaginationNext
											href="#"
											onClick={(e) => {
												e.preventDefault();
												setPage((p) => Math.min(totalPages, p + 1));
											}}
										/>
									</PaginationItem>
								</PaginationContent>
							</Pagination>
						</div>
					)}
				</CardContent>
			</Card>
		</div>
	);
}
