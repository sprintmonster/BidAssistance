import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Trash2 } from "lucide-react";

import { fetchWishlist, toggleWishlist, updateWishlist } from "../api/wishlist";
import type { WishlistItem } from "../types/wishlist";
import type { BidStage } from "../types/bid";
import { BID_STAGE_OPTIONS } from "../types/bid";

type SortKey = "DEADLINE_ASC" | "DEADLINE_DESC" | "TITLE_ASC";

type PastResult = "NONE" | "WON" | "LOST";

const ACTIVE_STAGES: BidStage[] = [
	"INTEREST",
	"REVIEW",
	"DECIDED",
	"DOC_PREP",
	"SUBMITTED",
];

const ACTIVE_STAGE_OPTIONS = BID_STAGE_OPTIONS.filter((o) =>
	(ACTIVE_STAGES as readonly string[]).includes(o.value),
);

const PAST_RESULT_OPTIONS: Array<{ value: PastResult; label: string }> = [
	{ value: "NONE", label: "미투찰" },
	{ value: "WON", label: "낙찰" },
	{ value: "LOST", label: "탈락" },
];

function parse_time(v: string): number {
	if (!v) return 0;
	const t = Date.parse(v);
	if (Number.isFinite(t)) return t;
	return 0;
}

function is_past_bid(bidEnd: string, nowMs: number): boolean {
	const t = parse_time(String(bidEnd));
	if (t <= 0) return false;
	return t < nowMs;
}

function formatAmount(value: unknown): string {
	if (value == null || value === "") return "-";
	const n =
		typeof value === "number"
			? value
			: Number(String(value).replace(/[^\d.-]/g, ""));
	if (!Number.isFinite(n)) return "-";
	return n.toLocaleString("ko-KR");
}

function formatDateTimeOneLine(dateStr: string) {
	if (!dateStr) return "-";
	const d = new Date(dateStr);
	if (!Number.isFinite(d.getTime())) return "-";

	const date = d
		.toLocaleDateString("ko-KR", {
			year: "numeric",
			month: "2-digit",
			day: "2-digit",
		})
		.replace(/\.\s*/g, ".");

	const time = d.toLocaleTimeString("ko-KR", {
		hour: "2-digit",
		minute: "2-digit",
		hour12: false,
	});

	return `${date} ${time}`;
}

function stage_label(stage: BidStage): string {
	const found = BID_STAGE_OPTIONS.find((o) => o.value === stage);
	return found ? found.label : stage;
}

function get_user_id(): number | null {
	const raw = localStorage.getItem("userId");
	if (!raw) return null;
	const n = Number(raw);
	if (!Number.isFinite(n)) return null;
	return n;
}

function days_until(dateStr: string, nowMs: number): number | null {
	if (!dateStr) return null;
	const end = new Date(dateStr);
	if (!Number.isFinite(end.getTime())) return null;
	const diff = end.getTime() - nowMs;
	if (!Number.isFinite(diff)) return null;
	if (diff < 0) return -1;
	return Math.floor(diff / 86400000);
}

function dday_label(daysLeft: number): string | null {
	if (!Number.isFinite(daysLeft)) return null;
	if (daysLeft < 0) return null;
	if (daysLeft === 0) return "D-DAY";
	return `D-${daysLeft}`;
}

function past_result_from_stage(stage: BidStage): PastResult {
	if (stage === "WON") return "WON";
	if (stage === "LOST") return "LOST";
	return "NONE";
}

export function CartPage({
	setGlobalLoading,
	showToast,
}: {
	setGlobalLoading: (v: boolean) => void;
	showToast: (msg: string, type: "success" | "error") => void;
}) {
	const navigate = useNavigate();

	const [wishlist, setWishlist] = useState<WishlistItem[]>([]);
	const [activeStage, setActiveStage] = useState<BidStage | "ALL">("ALL");
	const [sortKey, setSortKey] = useState<SortKey>("DEADLINE_ASC");
	const [nowMs, setNowMs] = useState(() => Date.now());

	useEffect(() => {
		const t = window.setInterval(() => setNowMs(Date.now()), 30000);
		return () => window.clearInterval(t);
	}, []);

	const loadWishlist = async () => {
		const userId = get_user_id();
		if (userId === null) {
			showToast("userId가 없습니다. 다시 로그인 해주세요.", "error");
			return;
		}
		const items = await fetchWishlist(userId);
		setWishlist(items);
	};

	useEffect(() => {
		void loadWishlist();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	const activeWishlist = useMemo(
		() => wishlist.filter((it) => !is_past_bid(String(it.bidEnd), nowMs)),
		[wishlist, nowMs],
	);

	const pastWishlist = useMemo(
		() => wishlist.filter((it) => is_past_bid(String(it.bidEnd), nowMs)),
		[wishlist, nowMs],
	);

	const stageCounts = useMemo(() => {
		const counts: Record<BidStage, number> = {
			INTEREST: 0,
			REVIEW: 0,
			DECIDED: 0,
			DOC_PREP: 0,
			SUBMITTED: 0,
			WON: 0,
			LOST: 0,
		};
		for (const it of activeWishlist) {
			if ((ACTIVE_STAGES as readonly string[]).includes(it.stage))
				counts[it.stage] = (counts[it.stage] ?? 0) + 1;
		}
		return counts;
	}, [activeWishlist]);

	const visibleItems = useMemo(() => {
		let items = activeWishlist.slice();

		if (activeStage !== "ALL") items = items.filter((it) => it.stage === activeStage);

		items.sort((a, b) => {
			if (sortKey === "TITLE_ASC")
				return String(a.title).localeCompare(String(b.title));
			const ta = parse_time(String(a.bidEnd));
			const tb = parse_time(String(b.bidEnd));
			if (sortKey === "DEADLINE_DESC") return tb - ta;
			return ta - tb;
		});

		return items;
	}, [activeWishlist, activeStage, sortKey]);

	const pastGrouped = useMemo(() => {
		const none: WishlistItem[] = [];
		const won: WishlistItem[] = [];
		const lost: WishlistItem[] = [];

		for (const it of pastWishlist) {
			const r = past_result_from_stage(it.stage);
			if (r === "WON") won.push(it);
			else if (r === "LOST") lost.push(it);
			else none.push(it);
		}

		const by_end_desc = (a: WishlistItem, b: WishlistItem) =>
			parse_time(String(b.bidEnd)) - parse_time(String(a.bidEnd));

		none.sort(by_end_desc);
		won.sort(by_end_desc);
		lost.sort(by_end_desc);

		return { none, won, lost };
	}, [pastWishlist]);

	const onDelete = async (bidId: number) => {
		try {
			const userId = get_user_id();
			if (userId === null) {
				showToast("userId가 없습니다. 다시 로그인 해주세요.", "error");
				return;
			}
			setGlobalLoading(true);
			const res = await toggleWishlist(userId, bidId);
			await loadWishlist();
			showToast(res.message || "삭제되었습니다.", "success");
		} catch (e: any) {
			showToast(e?.message || "삭제 실패", "error");
		} finally {
			setGlobalLoading(false);
		}
	};

	const onChangeStage = async (item: WishlistItem, stage: BidStage) => {
		try {
			const userId = get_user_id();
			if (userId === null) {
				showToast("userId가 없습니다. 다시 로그인 해주세요.", "error");
				return;
			}
			setGlobalLoading(true);

			const res = await updateWishlist({
				userId,
				bidId: item.bidId,
				wishlistId: item.id,
				stage,
			});

			setWishlist((prev) =>
				prev.map((it) => (it.bidId === item.bidId ? { ...it, stage } : it)),
			);

			showToast(res.message || "단계가 변경되었습니다.", "success");
		} catch (e: any) {
			showToast(e?.message || "단계 변경 실패", "error");
		} finally {
			setGlobalLoading(false);
		}
	};

	const onChangePastResult = async (item: WishlistItem, result: PastResult) => {
		const current = past_result_from_stage(item.stage);
		if (current === result) return;

		let nextStage: BidStage = item.stage;

		if (result === "WON") nextStage = "WON";
		else if (result === "LOST") nextStage = "LOST";
		else {
			nextStage = "SUBMITTED";
		}

		try {
			const userId = get_user_id();
			if (userId === null) {
				showToast("userId가 없습니다. 다시 로그인 해주세요.", "error");
				return;
			}
			setGlobalLoading(true);

			const res = await updateWishlist({
				userId,
				bidId: item.bidId,
				wishlistId: item.id,
				stage: nextStage,
			});

			setWishlist((prev) =>
				prev.map((it) =>
					it.bidId === item.bidId ? { ...it, stage: nextStage } : it,
				),
			);

			showToast(res.message || "지난 공고 상태가 변경되었습니다.", "success");
		} catch (e: any) {
			showToast(e?.message || "상태 변경 실패", "error");
		} finally {
			setGlobalLoading(false);
		}
	};

	function PastSection({
		title,
		count,
		items,
		badgeClass,
	}: {
		title: string;
		count: number;
		items: WishlistItem[];
		badgeClass: string;
	}) {
		return (
			<div className="border rounded-xl overflow-hidden">
				<div className="flex items-center justify-between px-4 py-3 bg-slate-50 border-b">
					<div className="flex items-center gap-2">
						<span className="font-semibold text-slate-900">{title}</span>
						<span
							className={[
								"inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold",
								badgeClass,
							].join(" ")}
						>
							{count}건
						</span>
					</div>
				</div>

				{items.length === 0 ? (
					<div className="px-4 py-5 text-sm text-slate-500">해당 항목이 없습니다.</div>
				) : (
					<div className="divide-y">
						{items.map((w) => {
							const amountText = w.baseAmount ? `${formatAmount(w.baseAmount)}원` : "";
							const endText = w.bidEnd
								? `마감 ${formatDateTimeOneLine(String(w.bidEnd))}`
								: "";
							const value = past_result_from_stage(w.stage);

							return (
								<div
									key={`past:${w.id}:${w.bidId}`}
									className="px-4 sm:px-5 py-4 hover:bg-slate-50 transition-colors"
								>
									<div className="grid grid-cols-1 sm:grid-cols-[minmax(0,1fr)_auto] gap-3 sm:gap-4 items-start">
										<button
											type="button"
											className="text-left min-w-0"
											onClick={() => navigate(`/bids/${w.bidId}`)}
										>
											<div className="font-semibold text-slate-900 truncate">
												{w.title}
											</div>

											<div className="mt-1 text-sm text-slate-500 flex flex-wrap gap-x-2 gap-y-1">
												<span>{w.agency}</span>
												{w.baseAmount ? (
													<>
														<span className="text-slate-300">·</span>
														<span>{amountText}</span>
													</>
												) : null}
												{w.bidEnd ? (
													<>
														<span className="text-slate-300">·</span>
														<span>{endText}</span>
													</>
												) : null}
											</div>
										</button>

										<div className="flex items-center justify-end gap-2">
											<select
												className="h-9 rounded-full border bg-white px-3 text-sm"
												value={value}
												onClick={(e) => e.stopPropagation()}
												onChange={(e) => {
													e.stopPropagation();
													void onChangePastResult(w, e.target.value as PastResult);
												}}
											>
												{PAST_RESULT_OPTIONS.map((opt) => (
													<option key={opt.value} value={opt.value}>
														{opt.label}
													</option>
												))}
											</select>

											<button
												type="button"
												className="h-9 w-9 inline-flex items-center justify-center rounded-full border bg-white hover:bg-red-50 hover:border-red-200 transition-colors"
												onClick={(e) => {
													e.stopPropagation();
													void onDelete(w.bidId);
												}}
												aria-label="삭제"
												title="삭제"
											>
												<Trash2 className="h-4 w-4 text-red-500" />
											</button>
										</div>
									</div>
								</div>
							);
						})}
					</div>
				)}
			</div>
		);
	}

	return (
		<div className="w-full">
			<div className="mx-auto w-full max-w-6xl px-4 sm:px-6 lg:px-8 space-y-5">
				<div>
					<h2 className="text-2xl font-bold">장바구니</h2>
					<div className="text-sm text-slate-500">장바구니에 담은 공고를 관리하세요</div>
				</div>

				<div className="bg-white border rounded-2xl p-4 sm:p-5">
					<div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
						{ACTIVE_STAGES.map((st) => {
							const label = stage_label(st);
							const count = stageCounts[st] ?? 0;
							const active = activeStage === st;
							return (
								<button
									key={st}
									type="button"
									onClick={() => setActiveStage(active ? "ALL" : st)}
									className={[
										"rounded-xl px-3 py-2 border text-left hover:bg-slate-50 transition-colors",
										active
											? "border-slate-900 bg-slate-50"
											: "border-slate-200 bg-white",
									].join(" ")}
								>
									<div className="text-xs text-slate-500">{label}</div>
									<div className="text-lg font-semibold text-slate-900">{count}건</div>
								</button>
							);
						})}
					</div>

					{activeStage !== "ALL" && (
						<div className="mt-3 text-sm text-slate-600 flex items-center gap-2">
							<span>
								필터 적용됨:{" "}
								<span className="font-semibold">{stage_label(activeStage)}</span>
							</span>
							<button
								type="button"
								onClick={() => setActiveStage("ALL")}
								className="text-blue-600 hover:underline"
							>
								해제
							</button>
						</div>
					)}
				</div>

				<div className="bg-white border rounded-2xl overflow-hidden">
					<div className="flex items-center justify-between gap-3 px-4 sm:px-5 py-3 border-b">
						<div className="font-semibold text-slate-900">장바구니 공고 목록</div>

						<div className="flex items-center gap-2">
							<span className="hidden sm:inline text-sm text-slate-500">정렬</span>
							<select
								className="h-9 rounded-lg border bg-white px-3 text-sm shadow-sm"
								value={sortKey}
								onChange={(e) => setSortKey(e.target.value as SortKey)}
							>
								<option value="DEADLINE_ASC">마감 빠른순</option>
								<option value="DEADLINE_DESC">마감 늦은순</option>
								<option value="TITLE_ASC">제목순</option>
							</select>
						</div>
					</div>

					{wishlist.length === 0 ? (
						<div className="p-5 text-slate-500">찜한 공고가 없습니다.</div>
					) : visibleItems.length === 0 ? (
						<div className="p-5 text-slate-500">
							{activeWishlist.length === 0
								? "현재 진행 중인 공고가 없습니다."
								: "해당 단계에 공고가 없습니다."}
						</div>
					) : (
						<div className="divide-y">
							{visibleItems.map((w) => {
								const amountText = w.baseAmount ? `${formatAmount(w.baseAmount)}원` : "";
								const endText = w.bidEnd ? `마감 ${formatDateTimeOneLine(String(w.bidEnd))}` : "";

								const daysLeft = w.bidEnd ? days_until(String(w.bidEnd), nowMs) : null;
								const label = daysLeft == null ? null : dday_label(daysLeft);
								const showBadge = label != null && daysLeft != null && daysLeft <= 7;

								const currentInActive =
									(ACTIVE_STAGES as readonly string[]).includes(w.stage);

								return (
									<div
										key={`${w.id}:${w.bidId}`}
										className="px-4 sm:px-5 py-4 hover:bg-slate-50 transition-colors"
									>
										<div className="grid grid-cols-1 sm:grid-cols-[minmax(0,1fr)_auto] gap-3 sm:gap-4 items-start">
											<button
												type="button"
												className="text-left min-w-0"
												onClick={() => navigate(`/bids/${w.bidId}`)}
											>
												<div className="flex items-center gap-2 min-w-0">
													<div className="font-semibold text-slate-900 truncate">
														{w.title}
													</div>
													{showBadge ? (
														<span
															className={[
																"shrink-0 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold",
																daysLeft === 0
																	? "bg-rose-100 text-rose-700"
																	: daysLeft <= 3
																		? "bg-amber-100 text-amber-800"
																		: "bg-slate-100 text-slate-700",
															].join(" ")}
														>
															{label}
														</span>
													) : null}
												</div>

												<div className="mt-1 text-sm text-slate-500 flex flex-wrap gap-x-2 gap-y-1">
													<span>{w.agency}</span>
													{w.baseAmount ? (
														<>
															<span className="text-slate-300">·</span>
															<span>{amountText}</span>
														</>
													) : null}
													{w.bidEnd ? (
														<>
															<span className="text-slate-300">·</span>
															<span>{endText}</span>
														</>
													) : null}
												</div>
											</button>

											<div className="flex items-center justify-end">
												<div className="inline-flex items-center gap-2 rounded-full border bg-slate-50 px-2 py-1 shadow-sm">
													<div className="w-[128px] sm:w-[140px]">
														<select
															className="h-9 w-full rounded-full border bg-white px-3 text-sm"
															value={currentInActive ? w.stage : "SUBMITTED"}
															onClick={(e) => e.stopPropagation()}
															onChange={(e) => {
																e.stopPropagation();
																void onChangeStage(w, e.target.value as BidStage);
															}}
														>
															{currentInActive ? null : (
																<option value="SUBMITTED">
																	{`(현재값: ${stage_label(w.stage)})`}
																</option>
															)}
															{ACTIVE_STAGE_OPTIONS.map((opt) => (
																<option key={opt.value} value={opt.value}>
																	{opt.label}
																</option>
															))}
														</select>
													</div>

													<button
														type="button"
														className="h-9 w-9 inline-flex items-center justify-center rounded-full border bg-white hover:bg-red-50 hover:border-red-200 transition-colors"
														onClick={(e) => {
															e.stopPropagation();
															void onDelete(w.bidId);
														}}
														aria-label="삭제"
														title="삭제"
													>
														<Trash2 className="h-4 w-4 text-red-500" />
													</button>
												</div>
											</div>
										</div>
									</div>
								);
							})}
						</div>
					)}
				</div>

				{pastWishlist.length > 0 ? (
					<div className="bg-white border rounded-2xl overflow-hidden">
						<div className="flex items-center justify-between gap-3 px-4 sm:px-5 py-3 border-b">
							<div className="font-semibold text-slate-900">지난 공고 목록</div>
							<div className="text-sm text-slate-500">{pastWishlist.length}건</div>
						</div>

						<div className="p-4 sm:p-5 space-y-4">
							<PastSection
								title="미투찰"
								count={pastGrouped.none.length}
								items={pastGrouped.none}
								badgeClass="bg-slate-100 text-slate-700"
							/>
							<PastSection
								title="낙찰"
								count={pastGrouped.won.length}
								items={pastGrouped.won}
								badgeClass="bg-emerald-100 text-emerald-700"
							/>
							<PastSection
								title="탈락"
								count={pastGrouped.lost.length}
								items={pastGrouped.lost}
								badgeClass="bg-rose-100 text-rose-700"
							/>
						</div>
					</div>
				) : null}
			</div>
		</div>
	);
}
