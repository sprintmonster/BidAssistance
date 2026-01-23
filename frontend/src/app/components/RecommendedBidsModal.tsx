import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Star } from "lucide-react";

import { fetchBids, type Bid } from "../api/bids";
import { toggleWishlist } from "../api/wishlist";

import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";

const SUPPRESS_KEY = "reco_popup_suppress_date";
const TRIGGER_KEY = "reco_popup_trigger";

function today_ymd(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function parse_date(s: string): Date | null {
  if (!s) return null;
  const d = new Date(s);
  if (!Number.isFinite(d.getTime())) return null;
  return d;
}

function days_left(endDate: string): number | null {
  const d = parse_date(endDate);
  if (!d) return null;

  const now = new Date();
  const diff = d.getTime() - now.getTime();
  const days = Math.ceil(diff / (1000 * 60 * 60 * 24));
  return days;
}

function format_krw_eok(raw: unknown): string {
  let n = 0;
  if (typeof raw === "number" && Number.isFinite(raw)) n = raw;
  else if (typeof raw === "bigint") n = Number(raw);
  else if (typeof raw === "string") {
    const digits = raw.replace(/[^0-9]/g, "");
    if (digits) n = Number(digits);
  }

  if (!Number.isFinite(n) || n <= 0) return "-";

  const eok = n / 100_000_000;
  const rounded = Math.round(eok * 10) / 10;
  const s = String(rounded);
  const cleaned = s.endsWith(".0") ? s.slice(0, -2) : s;
  return `${cleaned}억 원`;
}

function pick_bid_id(b: Bid): number {
  if (typeof b.id === "number" && Number.isFinite(b.id)) return b.id;
  if (typeof b.bidId === "number" && Number.isFinite(b.bidId)) return b.bidId;
  return 0;
}

function format_date_ymd(s: string): string {
  const d = parse_date(s);
  if (!d) return "-";
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export function suppress_reco_popup_today() {
  localStorage.setItem(SUPPRESS_KEY, today_ymd());
}

export function is_reco_popup_suppressed_today(): boolean {
  return localStorage.getItem(SUPPRESS_KEY) === today_ymd();
}

export function mark_reco_popup_trigger() {
  localStorage.setItem(TRIGGER_KEY, "1");
}

export function consume_reco_popup_trigger(): boolean {
  if (localStorage.getItem(TRIGGER_KEY) === "1") {
    localStorage.removeItem(TRIGGER_KEY);
    return true;
  }
  return false;
}

export function RecommendedBidsModal({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const navigate = useNavigate();

  const [bids, setBids] = useState<Bid[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [dontShowToday, setDontShowToday] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [togglingId, setTogglingId] = useState<number | null>(null);

  useEffect(() => {
    if (!open) return;

    let ignore = false;
    setNotice(null);
    setError(null);

    const load = async () => {
      setLoading(true);
      try {
        const list = await fetchBids();
        if (ignore) return;
        setBids(Array.isArray(list) ? list : []);
      } catch (e) {
        if (ignore) return;
        setBids([]);
        setError(e instanceof Error ? e.message : "추천 공고를 불러오지 못했습니다.");
      } finally {
        if (!ignore) setLoading(false);
      }
    };

    void load();
    return () => {
      ignore = true;
    };
  }, [open]);

  const recommended = useMemo(() => {
    const items = [...bids];
    items.sort((a, b) => {
      const ad =
        parse_date(String(a.endDate ?? ""))?.getTime() ?? Number.MAX_SAFE_INTEGER;
      const bd =
        parse_date(String(b.endDate ?? ""))?.getTime() ?? Number.MAX_SAFE_INTEGER;
      return ad - bd;
    });
    return items.slice(0, 10);
  }, [bids]);

  const on_close = () => {
    if (dontShowToday) suppress_reco_popup_today();
    onOpenChange(false);
  };

  const on_detail = (bid: Bid) => {
    const id = pick_bid_id(bid);
    if (!id) return;
    onOpenChange(false);
    navigate(`/bids/${id}`);
  };

  const on_toggle_wishlist = async (bid: Bid) => {
    const raw = localStorage.getItem("userId");
    const userId = raw ? Number(raw) : NaN;
    if (!Number.isFinite(userId)) {
      setNotice("로그인이 필요합니다.");
      return;
    }

    const bidId = pick_bid_id(bid);
    if (!bidId) return;

    try {
      setNotice(null);
      setTogglingId(bidId);
      const res = await toggleWishlist(userId, bidId);
      if (res.status === "success") setNotice("장바구니에 반영했습니다.");
      else setNotice(res.message || "장바구니 처리에 실패했습니다.");
    } catch (e) {
      setNotice(e instanceof Error ? e.message : "장바구니 처리 중 오류가 발생했습니다.");
    } finally {
      setTogglingId(null);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) {
          on_close();
          return;
        }
        onOpenChange(true);
      }}
    >
      <DialogContent className="p-0 sm:max-w-2xl rounded-2xl overflow-hidden">
        <DialogHeader className="px-6 pt-6">
          <DialogTitle className="flex items-center gap-2 text-lg font-semibold">
            <Star className="w-5 h-5" />
            추천 공고
          </DialogTitle>
        </DialogHeader>

        <div className="px-6 pb-4">
          <div className="rounded-2xl border bg-blue-50/40 px-5 py-4">
            <div className="font-semibold text-slate-900">맞춤 추천 공고</div>
            <div className="text-sm text-slate-500 mt-1">
              관심 지역/금액/최근 조회 흐름을 기반으로 추천됩니다. (현재는 데모 데이터)
            </div>
          </div>

          {notice ? <div className="mt-3 text-sm text-slate-600">{notice}</div> : null}

          <div className="mt-4">
            <div className="max-h-[420px] overflow-y-auto pr-1">
              {loading ? (
                <div className="space-y-3">
                  {Array.from({ length: 4 }).map((_, idx) => (
                    <div key={idx} className="border rounded-2xl p-4 bg-white">
                      <div className="h-4 w-48 bg-slate-100 rounded animate-pulse" />
                      <div className="mt-2 h-3 w-36 bg-slate-100 rounded animate-pulse" />
                      <div className="mt-4 h-8 w-44 bg-slate-100 rounded animate-pulse" />
                    </div>
                  ))}
                </div>
              ) : error ? (
                <div className="border rounded-2xl p-4 bg-red-50 text-red-700 text-sm">
                  {error}
                </div>
              ) : recommended.length === 0 ? (
                <div className="border rounded-2xl p-6 bg-white text-sm text-slate-500">
                  표시할 추천 공고가 없습니다.
                </div>
              ) : (
                <div className="space-y-3">
                  {recommended.map((b) => {
                    const dleft = days_left(String(b.endDate ?? ""));
                    const dText = dleft === null ? "-" : dleft <= 0 ? "마감" : `D-${dleft}`;

                    return (
                      <div
                        key={String(pick_bid_id(b) || b.realId)}
                        className="border rounded-2xl p-4 bg-white"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <div className="font-semibold text-slate-900 truncate">
                              {b.name}
                            </div>
                            <div className="text-sm text-slate-500 truncate">
                              {b.organization}
                            </div>
                          </div>

                          <div className="flex items-center gap-2 shrink-0">
                            <span className="px-2 py-1 rounded-full text-xs border bg-white">
                              {b.region || "-"}
                            </span>
                            <span
                              className={[
                                "px-2 py-1 rounded-full text-xs font-semibold",
                                dleft !== null && dleft > 0
                                  ? "bg-blue-600 text-white"
                                  : "bg-slate-100 text-slate-600",
                              ].join(" ")}
                            >
                              {dText}
                            </span>
                          </div>
                        </div>

                        <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-600">
                          <span className="px-2 py-1 rounded-full bg-slate-50 border">
                            {format_krw_eok((b as any).estimatePrice)}
                          </span>
                          <span className="px-2 py-1 rounded-full bg-slate-50 border">
                            마감: {format_date_ymd(String(b.endDate ?? ""))}
                          </span>
                        </div>

                        <div className="mt-4 flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => on_detail(b)}
                            className="h-9 px-3 rounded-xl bg-slate-900 text-white hover:bg-slate-800 text-sm"
                          >
                            상세 보기
                          </button>
                          <button
                            type="button"
                            onClick={() => void on_toggle_wishlist(b)}
                            disabled={togglingId === pick_bid_id(b)}
                            className="h-9 px-3 rounded-xl border hover:bg-slate-50 text-sm disabled:opacity-60"
                          >
                            장바구니 담기
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 flex items-center justify-between">
            <label className="flex items-center gap-2 text-sm text-slate-600 select-none">
              <input
                type="checkbox"
                checked={dontShowToday}
                onChange={(e) => setDontShowToday(e.target.checked)}
              />
              오늘 다시 보지 않기
            </label>

            <button
              type="button"
              onClick={on_close}
              className="h-9 px-4 rounded-xl border hover:bg-slate-50 text-sm"
            >
              닫기
            </button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
