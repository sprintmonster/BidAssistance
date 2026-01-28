import { useEffect, useMemo, useState } from "react";

import { fetchBids, type Bid } from "../api/bids";
import { fetchWishlist } from "../api/wishlist";
import type { WishlistItem } from "../types/wishlist";

import { SummaryCards } from "./dashboard/SummaryCard";
import { MonthlyTrendChart, type MonthlyTrendPoint } from "./dashboard/MonthlyTrendChart";
import { RegionPieChart, type RegionDistPoint } from "./dashboard/RegionPieChart";

type Kpi = {

  newBidsThisMonth: number;

  wishlistCount: number;
  closingSoon3Days: number;
  totalExpectedAmountEok: string;
};

export function Dashboard() {
  const [bids, setBids] = useState<Bid[]>([]);
  const [wishlist, setWishlist] = useState<WishlistItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);

      try {
        const res = await fetchBids();
        const items = pick_list(res);
        setBids(items as Bid[]);
      } catch (e) {
        setBids([]);
        setError(e instanceof Error ? e.message : "공고 데이터를 불러오지 못했습니다.");
      } finally {
        setLoading(false);
      }

      const userIdStr = localStorage.getItem("userId");
      const userId = Number(userIdStr);
      if (!userIdStr || !Number.isFinite(userId)) {
        setWishlist([]);
        return;
      }

      try {
        const w = await fetchWishlist(userId);
        setWishlist(w);
      } catch {
        setWishlist([]);
      }
    };

    void load();
  }, []);

  const monthlyTrend = useMemo<MonthlyTrendPoint[]>(
    () => build_monthly_trend_forward(bids, 6),
    [bids],
  );

  const regionDist = useMemo<RegionDistPoint[]>(() => build_region_dist(bids), [bids]);
  const kpi = useMemo<Kpi>(() => build_kpi(bids, wishlist), [bids, wishlist]);

  return (
    <div className="space-y-8">
      {error ? (
        <div className="border rounded-2xl p-4 bg-red-50 text-red-700 text-sm">
          {error}
        </div>
      ) : null}

      <SummaryCards loading={loading} kpi={kpi} />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <MonthlyTrendChart loading={loading} data={monthlyTrend} />
        <RegionPieChart loading={loading} data={regionDist} />
      </div>
    </div>
  );
}

function pick_list(res: unknown): unknown[] {
  if (Array.isArray(res)) return res;
  const anyRes = res as any;
  if (Array.isArray(anyRes?.data)) return anyRes.data;
  if (Array.isArray(anyRes?.data?.items)) return anyRes.data.items;
  if (Array.isArray(anyRes?.data?.content)) return anyRes.data.content;
  return [];
}

function to_date(s: string): Date | null {
  if (!s) return null;
  const d = new Date(s);
  if (!Number.isFinite(d.getTime())) return null;
  return d;
}

function month_key(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  return `${y}-${m}`;
}

function month_label(d: Date): string {
  const yy = String(d.getFullYear()).slice(2);
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  return `${yy}.${mm}`;
}

function add_months(d: Date, months: number): Date {
  return new Date(d.getFullYear(), d.getMonth() + months, 1);
}

function build_month_scaffold_forward(n: number): Array<{ key: string; label: string; start: Date; end: Date }> {
  const now = new Date();
  const base = new Date(now.getFullYear(), now.getMonth(), 1);

  const out: Array<{ key: string; label: string; start: Date; end: Date }> = [];
  for (let i = 0; i < n; i += 1) {
    const start = add_months(base, i);
    const end = add_months(base, i + 1);
    out.push({
      key: month_key(start),
      label: month_label(start),
      start,
      end,
    });
  }
  return out;
}

export function build_monthly_trend_forward(bids: Bid[], n: number): MonthlyTrendPoint[] {
  const scaffold = build_month_scaffold_forward(n);
  const counts = new Map<string, number>();

  bids.forEach((b) => {
    const d = to_date(String((b as any).endDate ?? (b as any).bidEnd ?? ""));
    if (!d) return;

    for (const m of scaffold) {
      if (d.getTime() >= m.start.getTime() && d.getTime() < m.end.getTime()) {
        counts.set(m.key, (counts.get(m.key) ?? 0) + 1);
        break;
      }
    }
  });

  return scaffold.map((m) => ({ month: m.label, value: counts.get(m.key) ?? 0 }));
}

function normalize_region(raw: string): string {
	const v = (raw || "").trim();
	if (!v) return "기타";

	const rules: Array<[string[], string]> = [
		[["전국"], "전국"],

		[["서울", "서울특별시"], "서울특별시"],
		[["부산", "부산광역시"], "부산광역시"],
		[["대구", "대구광역시"], "대구광역시"],
		[["인천", "인천광역시"], "인천광역시"],
		[["광주", "광주광역시"], "광주광역시"],
		[["대전", "대전광역시"], "대전광역시"],
		[["울산", "울산광역시"], "울산광역시"],
		[["세종", "세종특별자치시"], "세종특별자치시"],

		[["경기", "경기도"], "경기도"],
		[["강원", "강원도", "강원특별자치도"], "강원특별자치도"],
		[["충북", "충청북도"], "충청북도"],
		[["충남", "충청남도"], "충청남도"],
		[["전북", "전라북도", "전북특별자치도"], "전북특별자치도"],
		[["전남", "전라남도"], "전라남도"],
		[["경북", "경상북도"], "경상북도"],
		[["경남", "경상남도"], "경상남도"],
		[["제주", "제주도", "제주특별자치도"], "제주특별자치도"],
	];

	for (const [needles, label] of rules) {
		for (const needle of needles) {
			if (v.includes(needle)) return label;
		}
	}

	return v.split(/[\s/,(]/)[0] || "기타";
}

export function build_region_dist(bids: Bid[]): RegionDistPoint[] {
	const counts = new Map<string, number>();

	bids.forEach((b) => {
		const region = normalize_region(String((b as any).region ?? ""));
		counts.set(region, (counts.get(region) ?? 0) + 1);
	});

	return Array.from(counts.entries())
		.map(([name, value]) => ({ name, value }))
		.filter((x) => x.value > 0)
		.sort((a, b) => b.value - a.value);
}


function is_same_month(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth();
}

function add_days(d: Date, days: number): Date {
  const x = new Date(d);
  x.setDate(x.getDate() + days);
  return x;
}

function parse_amount(v: unknown): number {
  if (typeof v === "number") return Number.isFinite(v) ? v : 0;
  if (typeof v !== "string") return 0;
  const digits = v.replace(/[^0-9]/g, "");
  if (!digits) return 0;
  const n = Number(digits);
  return Number.isFinite(n) ? n : 0;
}

function format_eok(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0";
  const eok = value / 100_000_000;
  const rounded = Math.round(eok * 10) / 10;
  const s = String(rounded);
  return s.endsWith(".0") ? s.slice(0, -2) : s;
}
function start_of_today(d: Date): Date {
    const x = new Date(d);
    x.setHours(0, 0, 0, 0);
    return x;
}

function end_of_today(d: Date): Date {
    const x = new Date(d);
    x.setHours(23, 59, 59, 999);
    return x;
}


function build_kpi(bids: Bid[], wishlist: WishlistItem[]): Kpi {
    const now = new Date();
    const threeDaysLater = add_days(now, 3);

    const todayStart = start_of_today(now);
    const todayEnd = end_of_today(now);

    let newBidsThisMonth = 0;   // (이름 유지할 거면 그대로)
    let closingSoon3Days = 0;

    bids.forEach((b) => {
        const start = to_date(String((b as any).startDate ?? (b as any).bidStart ?? ""));
        const end = to_date(String((b as any).endDate ?? (b as any).bidEnd ?? ""));

        // 오늘 시작한 공고로 변경
        if (start && start.getTime() >= todayStart.getTime() && start.getTime() <= todayEnd.getTime()) {
            newBidsThisMonth += 1;
        }

        if (end && end.getTime() >= now.getTime() && end.getTime() <= threeDaysLater.getTime()) {
            closingSoon3Days += 1;
        }
    });

    const sumAmount = wishlist.reduce((acc, it) => acc + parse_amount(it.baseAmount), 0);

    return {
        newBidsThisMonth, // (혹은 newBidsToday로 이름 바꾸면 newBidsToday)
        wishlistCount: wishlist.length,
        closingSoon3Days,
        totalExpectedAmountEok: format_eok(sumAmount),
    };
}
