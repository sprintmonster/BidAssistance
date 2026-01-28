import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Trash2 } from "lucide-react";

import { fetchWishlist, toggleWishlist, updateWishlist } from "../api/wishlist";
import type { WishlistItem } from "../types/wishlist";
import type { BidStage } from "../types/bid";
import { BID_STAGE_OPTIONS, BID_STAGES } from "../types/bid";

type SortKey = "DEADLINE_ASC" | "DEADLINE_DESC" | "TITLE_ASC";

function parse_time(v: string): number {
	if (!v) return 0;
	const t = Date.parse(v);
	if (Number.isFinite(t)) return t;
	return 0;
}
function formatAmount(value: unknown): string {
    if (value == null || value === "") return "-";

    // "123,000" 같은 문자열도 처리
    const n =
        typeof value === "number"
            ? value
            : Number(String(value).replace(/[^\d.-]/g, ""));

    if (!Number.isFinite(n)) return "-";
    return n.toLocaleString("ko-KR");
}

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
        hour12: true, // 오전/오후
    });

    return { dateLine, timeLine };
}
function formatDateTimeOneLine(dateStr: string) {
    if (!dateStr) return "-";

    const d = new Date(dateStr);
    if (!Number.isFinite(d.getTime())) return "-";

    const date = d.toLocaleDateString("ko-KR", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
    }).replace(/\.\s*/g, ".");

    const time = d.toLocaleTimeString("ko-KR", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false, // 24시간제 → 오전/오후 제거
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
		for (const it of wishlist) {
			counts[it.stage] = (counts[it.stage] ?? 0) + 1;
		}
		return counts;
	}, [wishlist]);

	const visibleItems = useMemo(() => {
		let items = wishlist.slice();
		if (activeStage !== "ALL") {
			items = items.filter((it) => it.stage === activeStage);
		}

		items.sort((a, b) => {
			if (sortKey === "TITLE_ASC") return String(a.title).localeCompare(String(b.title));
			const ta = parse_time(String(a.bidEnd));
			const tb = parse_time(String(b.bidEnd));
			if (sortKey === "DEADLINE_DESC") return tb - ta;
			return ta - tb;
		});

		return items;
	}, [wishlist, activeStage, sortKey]);

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
		} catch {
			showToast("삭제 실패", "error");
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

			// 낙/탈/제출/결정 등 timestamp는 보통 백엔드가 처리하므로, 우선 stage만 프론트 반영
			setWishlist((prev) => prev.map((it) => (it.bidId === item.bidId ? { ...it, stage } : it)));
			showToast(res.message || "단계가 변경되었습니다.", "success");
		} catch (e: any) {
			showToast(e?.message || "단계 변경 실패", "error");
		} finally {
			setGlobalLoading(false);
		}
	};

	return (
		<div className="space-y-5">
			<div>
				<h2 className="text-2xl font-bold">장바구니</h2>
				<div className="text-sm text-slate-500">장바구니에 담은 공고를 관리하세요</div>
			</div>

			{/* 파이프라인 요약 */}
			<div className="bg-white border rounded-2xl p-4">
				<div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
					{BID_STAGES.map((st) => {
						const label = stage_label(st);
						const count = stageCounts[st] ?? 0;
						const active = activeStage === st;
						return (
							<button
								key={st}
								type="button"
								onClick={() => setActiveStage(active ? "ALL" : st)}
								className={[
									"rounded-xl px-3 py-2 border text-left hover:bg-slate-50",
									active ? "border-slate-900 bg-slate-50" : "border-slate-200",
								].join(" ")}
							>
								<div className="text-xs text-slate-500">{label}</div>
								<div className="text-lg font-semibold text-slate-900">{count}건</div>
							</button>
						);
					})}
				</div>
				{activeStage !== "ALL" && (
					<div className="mt-3 text-sm text-slate-600">
						필터 적용됨: <span className="font-semibold">{stage_label(activeStage)}</span>
						<button
							type="button"
							onClick={() => setActiveStage("ALL")}
							className="ml-2 text-blue-600 hover:underline"
						>
							해제
						</button>
					</div>
				)}
			</div>

			{/* 목록 */}
			<div className="bg-white border rounded-2xl overflow-hidden">
				<div className="flex items-center justify-between px-4 py-3 border-b">
					<div className="font-semibold text-slate-900">장바구니 공고 목록</div>
					<select
						className="h-9 rounded-lg border bg-white px-3 text-sm"
						value={sortKey}
						onChange={(e) => setSortKey(e.target.value as SortKey)}
					>
						<option value="DEADLINE_ASC">마감 빠른순</option>
						<option value="DEADLINE_DESC">마감 늦은순</option>
						<option value="TITLE_ASC">제목순</option>
					</select>
				</div>

				{wishlist.length === 0 ? (
					<div className="p-4 text-slate-500">찜한 공고가 없습니다.</div>
				) : visibleItems.length === 0 ? (
					<div className="p-4 text-slate-500">해당 단계에 공고가 없습니다.</div>
				) : (
					<div className="divide-y">
						{visibleItems.map((w) => (
							<div
								key={`${w.id}:${w.bidId}`}
								className="px-4 py-4 flex items-start justify-between gap-4 hover:bg-slate-50 cursor-pointer"
								onClick={() => navigate(`/bids/${w.bidId}`)}
								role="button"
								tabIndex={0}
							>
								<div className="min-w-0">
									<div className="font-semibold text-slate-900 truncate">{w.title}</div>
                                    <div className="mt-1 text-sm text-slate-500">
                                        {(() => {
                                            const amountText = w.baseAmount ? `${formatAmount(w.baseAmount)}원` : "";
                                            const { dateLine, timeLine } = w.bidEnd ? formatDateTimeLines(String(w.bidEnd)) : { dateLine: "", timeLine: "" };

                                            return (
                                                <>
                                                    <span>{w.agency}</span>

                                                    {w.baseAmount ? (
                                                        <>
                                                            <span>{` · `}</span>
                                                            <span>{amountText}</span>
                                                        </>
                                                    ) : null}

                                                    {w.bidEnd ? ` · 마감 ${formatDateTimeOneLine(String(w.bidEnd))}` : ""}

                                                </>
                                            );
                                        })()}
                                    </div>

                                </div>

								<div className="flex items-center gap-2 shrink-0">
									<select
										className="h-9 rounded-lg border bg-slate-50 px-3 text-sm"
										value={w.stage}
										onClick={(e) => e.stopPropagation()}
										onChange={(e) => {
											e.stopPropagation();
											void onChangeStage(w, e.target.value as BidStage);
										}}
									>
										{BID_STAGE_OPTIONS.map((opt) => (
											<option key={opt.value} value={opt.value}>
												{opt.label}
											</option>
										))}
									</select>

									<button
										type="button"
										className="h-9 w-9 inline-flex items-center justify-center rounded-lg border hover:bg-red-50 hover:border-red-200"
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
						))}
					</div>
				)}
			</div>
		</div>
	);
}
