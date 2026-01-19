import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { Eye, FilterX, Plus, RefreshCw, Search } from "lucide-react";

import { fetchBids, type Bid } from "../api/bids";
import { api } from "../api/client";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "./ui/card";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "./ui/dialog";
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

	const [bids, setBids] = useState<Bid[]>([]);
	const [keyword, setKeyword] = useState<string>(urlQuery);
	const [agency, setAgency] = useState<string>("all");
	const [sortKey, setSortKey] = useState<SortKey>("deadline_asc");
	const [page, setPage] = useState<number>(1);
	const [pageSize, setPageSize] = useState<number>(10);
	const [selected, setSelected] = useState<Bid | null>(null);
	const [addingId, setAddingId] = useState<number | null>(null);
	const [addedIds, setAddedIds] = useState<Set<number>>(() => new Set());

	useEffect(() => {
		setKeyword(urlQuery);
		setPage(1);
	}, [urlQuery]);

	const load = async () => {
		try {
			setGlobalLoading(true);
			const list = await fetchBids();
			setBids(list);
		} catch {
			showToast("공고 목록을 불러오지 못했습니다.", "error");
		} finally {
			setGlobalLoading(false);
		}
	};

	useEffect(() => {
		void load();
		// eslint-disable-next-line react-hooks/exhaustive-deps
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

		if (agency !== "all") {
			list = list.filter((b) => (b.agency || "").trim() === agency);
		}

		if (q) {
			list = list.filter((b) => {
				const hay = `${b.title} ${b.agency} ${b.budget} ${b.deadline}`.toLowerCase();
				return hay.includes(q);
			});
		}

		list.sort((a, b) => {
			if (sortKey === "title_asc") {
				return a.title.localeCompare(b.title);
			}

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
			setAddingId(bidId);
			setGlobalLoading(true);
			await api("/wishlist", {
				method: "POST",
				body: JSON.stringify({ bidId }),
			});
			setAddedIds((prev) => {
				const next = new Set(prev);
				next.add(bidId);
				return next;
			});
			showToast("장바구니에 추가됨", "success");
		} catch {
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

	return (
		<div className="space-y-4">
			<Card>
				<CardHeader className="space-y-1">
					<CardTitle className="text-xl">공고 찾기</CardTitle>
					<CardDescription>
						키워드/기관/정렬 기반으로 공고를 빠르게 좁히고, 장바구니에
						담아 관리하세요.
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
									placeholder={
										urlQuery
											? `검색어: ${urlQuery}`
											: "키워드 검색 (공고명/기관/예산/마감)"
									}
									className="pl-8"
								/>
							</div>
							<Button
								variant="outline"
								onClick={() => {
									void load();
								}}
								className="shrink-0"
							>
								<RefreshCw className="mr-2 size-4" />
								새로고침
							</Button>
							<Button
								variant="ghost"
								onClick={resetFilters}
								className="shrink-0"
							>
								<FilterX className="mr-2 size-4" />
								필터 초기화
							</Button>
						</div>

						<div className="flex flex-col gap-2 sm:flex-row sm:items-center">
							<div className="flex items-center gap-2">

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
							</div>

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

					<div className="rounded-lg border">
						<Table>
							<TableHeader>
								<TableRow>
									<TableHead className="w-[120px]">마감</TableHead>
									<TableHead>공고명</TableHead>
									<TableHead className="w-[220px]">발주기관</TableHead>
									<TableHead className="w-[160px] text-right">예산</TableHead>
									<TableHead className="w-[140px]">상태</TableHead>
									<TableHead className="w-[160px] text-right">액션</TableHead>
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
										const alreadyAdded = addedIds.has(b.id);

										return (
											<TableRow
												key={b.id}
												className="cursor-pointer"
												onClick={() => setSelected(b)}
											>
												<TableCell className="whitespace-normal">
													<div className="flex flex-col">
														<span className="text-sm font-medium">
															{dday || b.deadline}
														</span>
														{dday && (
															<span className="text-xs text-muted-foreground">
																{b.deadline}
															</span>
														)}
													</div>
												</TableCell>
												<TableCell className="whitespace-normal">
													<div className="line-clamp-2 font-medium">{b.title}</div>
												</TableCell>
												<TableCell className="whitespace-normal">
													<div className="line-clamp-2">{b.agency}</div>
												</TableCell>
												<TableCell className="text-right">
													{b.budget}
												</TableCell>
												<TableCell>
													<Badge variant={dday === "D-DAY" ? "destructive" : "secondary"}>
														{dday ? "마감일" : "확인 필요"}
													</Badge>
												</TableCell>
												<TableCell className="text-right">
													<div className="flex justify-end gap-2">
														<Button
															variant="outline"
															size="sm"
															onClick={(e) => {
																e.stopPropagation();
																setSelected(b);
															}}
														>
															<Eye className="mr-2 size-4" />
															상세
														</Button>
														<Button
															size="sm"
															disabled={addingId === b.id || alreadyAdded}
															className={cn(alreadyAdded && "opacity-70")}
															onClick={(e) => {
																e.stopPropagation();
																void addToCart(b.id);
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

			<Dialog
				open={!!selected}
				onOpenChange={(open) => {
					if (!open) setSelected(null);
				}}
			>
				<DialogContent className="sm:max-w-2xl">
					{selected && (
						<DialogHeader>
							<DialogTitle className="leading-snug">{selected.title}</DialogTitle>
							<DialogDescription>
								발주기관: {selected.agency} · 예산: {selected.budget} · 마감: {selected.deadline}
							</DialogDescription>
						</DialogHeader>
					)}

					{selected && (
						<div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2">
							<div className="rounded-lg border p-3">
								<div className="text-xs text-muted-foreground">마감</div>
								<div className="mt-1 flex items-center gap-2">
									<span className="text-sm font-medium">
										{formatDday(selected.deadline) || selected.deadline}
									</span>
									<Badge variant="secondary">공고</Badge>
								</div>
							</div>
							<div className="rounded-lg border p-3">
								<div className="text-xs text-muted-foreground">예산</div>
								<div className="mt-1 text-sm font-medium">{selected.budget}</div>
							</div>
							<div className="rounded-lg border p-3 sm:col-span-2">
								<div className="text-xs text-muted-foreground">기관</div>
								<div className="mt-1 text-sm font-medium">{selected.agency}</div>
							</div>
							<div className="sm:col-span-2 flex justify-end gap-2">
								<Button variant="outline" onClick={() => setSelected(null)}>
									닫기
								</Button>
								<Button
									disabled={addingId === selected.id || addedIds.has(selected.id)}
									onClick={() => void addToCart(selected.id)}
								>
									<Plus className="mr-2 size-4" />
									{addedIds.has(selected.id) ? "담김" : "트래킹에 담기"}
								</Button>
							</div>
						</div>
					)}
				</DialogContent>
			</Dialog>
		</div>
	);
}
