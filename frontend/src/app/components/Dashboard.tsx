import { useEffect, useMemo, useState } from "react";
import { fetchBids, type Bid } from "../api/bids";
import { fetchWishlist } from "../api/wishlist";
import type { WishlistItem } from "../types/wishlist";

import { SummaryCards } from "./dashboard/SummaryCard";
import { MonthlyTrendChart, type MonthlyTrendPoint } from "./dashboard/MonthlyTrendChart";
import { RegionPieChart, type RegionDistPoint } from "./dashboard/RegionPieChart";

/* ================= KPI 타입 ================= */

type Kpi = {
    newBidsToday: number;
    totalExpectedAmount: number;

    newBidsThisMonth: number;
    wishlistCount: number;
    closingSoon3Days: number;
    totalExpectedAmountEok: string;
};

const EMPTY_KPI: Kpi = {
    newBidsToday: 0,
    wishlistCount: 0,
    closingSoon3Days: 0,
    totalExpectedAmount: 0,
    newBidsThisMonth: 0,
    totalExpectedAmountEok: "0",
};

/* ================= 유틸 ================= */

function parseDateSafe(s: string): Date | null {
    const t = Date.parse(String(s ?? ""));
    return Number.isFinite(t) ? new Date(t) : null;
}

function startOfToday(d = new Date()): Date {
    const x = new Date(d);
    x.setHours(0, 0, 0, 0);
    return x;
}

function endOfToday(d = new Date()): Date {
    const x = new Date(d);
    x.setHours(23, 59, 59, 999);
    return x;
}

function add_days(d: Date, days: number): Date {
    const x = new Date(d);
    x.setDate(x.getDate() + days);
    return x;
}

function toNumberAmount(v: unknown): number {
    if (v == null || v === "") return 0;
    if (typeof v === "number") return Number.isFinite(v) ? v : 0;
    const n = Number(String(v).replace(/[^\d.-]/g, ""));
    return Number.isFinite(n) ? n : 0;
}

function is_same_month(a: Date, b: Date): boolean {
    return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth();
}

function format_eok(value: number): string {
    if (!Number.isFinite(value) || value <= 0) return "0";
    const eok = value / 100_000_000;
    return Math.round(eok * 10) / 10 + "";
}

/* ================= KPI 카드 ================= */

function KpiCard({
                     title,
                     value,
                     sub,
                     icon,
                     accent,
                 }: {
    title: string;
    value: string;
    sub: string;
    icon: string;
    accent?: "warn";
}) {
    return (
        <div className="border rounded-2xl p-5 bg-white flex items-start justify-between">
            <div className="space-y-3">
                <div className="text-sm text-gray-600">{title}</div>
                <div className="text-3xl font-bold">{value}</div>
                <div className="text-sm text-gray-500">{sub}</div>
            </div>
            <div
                className={[
                    "w-10 h-10 rounded-xl flex items-center justify-center text-lg",
                    accent === "warn" ? "bg-orange-50" : "bg-gray-50",
                ].join(" ")}
            >
        <span className={accent === "warn" ? "text-orange-600" : "text-gray-700"}>
          {icon}
        </span>
            </div>
        </div>
    );
}

/* ================= Dashboard ================= */

export function Dashboard() {
    const [bids, setBids] = useState<Bid[]>([]);
    const [wishlist, setWishlist] = useState<WishlistItem[]>([]);
    const [loading, setLoading] = useState(true);

    // 🔧 수정: 데이터 로딩 전용 useEffect
    useEffect(() => {
        const load = async () => {
            setLoading(true);

            try {
                const bidList = await fetchBids();
                setBids(bidList);
            } catch {
                setBids([]);
            }

            const uidStr = localStorage.getItem("userId");
            const userId = Number(uidStr);

            if (!uidStr || !Number.isFinite(userId)) {
                setWishlist([]);
                setLoading(false);
                return;
            }

            try {
                const w = await fetchWishlist(userId);
                setWishlist(w);
            } catch {
                setWishlist([]);
            } finally {
                setLoading(false);
            }
        };

        void load();
    }, []);

    // 🔧 수정: KPI + 차트 계산은 useMemo로 통합
    const monthlyTrend = useMemo<MonthlyTrendPoint[]>(() => build_monthly_trend(bids, 6), [bids]);
    const regionDist = useMemo<RegionDistPoint[]>(() => build_region_dist(bids), [bids]);
    const kpi = useMemo<Kpi>(() => build_kpi(bids, wishlist), [bids, wishlist]);

    return (
        <div className="space-y-6">
            {/* KPI */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <KpiCard title="신규 공고" value={`${kpi.newBidsToday}개`} sub="오늘 시작" icon="📄" />
                <KpiCard title="관심 공고" value={`${kpi.wishlistCount}개`} sub="장바구니" icon="📈" />
                <KpiCard
                    title="마감 임박"
                    value={`${kpi.closingSoon3Days}개`}
                    sub="3일 이내"
                    icon="⏰"
                    accent="warn"
                />
                <KpiCard
                    title="총 예상액"
                    value={`${Math.round(kpi.totalExpectedAmount / 100_000_000)}억`}
                    sub="관심 공고 합계"
                    icon="💰"
                />
            </div>

            {/*<SummaryCards loading={loading} kpi={kpi} />*/}

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <MonthlyTrendChart loading={loading} data={monthlyTrend} />
                <RegionPieChart loading={loading} data={regionDist} />
            </div>
        </div>
    );
}

/* ================= KPI 계산 ================= */

function build_kpi(bids: Bid[], wishlist: WishlistItem[]): Kpi {
    const now = new Date();
    const todayStart = startOfToday(now);
    const todayEnd = endOfToday(now);
    const threeDaysLater = add_days(now, 3);

    let newBidsToday = 0;
    let newBidsThisMonth = 0;
    let closingSoon3Days = 0;

    bids.forEach((b) => {
        const start = parseDateSafe((b as any).startDate ?? (b as any).bidStart ?? "");
        const end = parseDateSafe((b as any).endDate ?? (b as any).bidEnd ?? "");

        if (start && start >= todayStart && start <= todayEnd) newBidsToday++;
        if (start && is_same_month(start, now)) newBidsThisMonth++;

        if (end && end >= now && end <= threeDaysLater) closingSoon3Days++;
    });

    const totalExpectedAmount = wishlist.reduce(
        (acc, it) => acc + toNumberAmount(it.baseAmount),
        0
    );

    return {
        newBidsToday,
        newBidsThisMonth,
        wishlistCount: wishlist.length,
        closingSoon3Days,
        totalExpectedAmount,
        totalExpectedAmountEok: format_eok(totalExpectedAmount),
    };
}

/* ================= 차트 계산 ================= */

function to_date(s: string): Date | null {
    if (!s) return null;
    const d = new Date(s);
    if (!Number.isFinite(d.getTime())) return null;
    return d;
}

function month_key(d: Date): string {
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
}

function month_label(d: Date): string {
    return `${String(d.getFullYear()).slice(2)}.${String(d.getMonth() + 1).padStart(2, "0")}`;
}

function build_month_scaffold(n: number): Array<{ key: string; label: string }> {
    const now = new Date();
    return Array.from({ length: n }, (_, i) => {
        const d = new Date(now.getFullYear(), now.getMonth() - (n - 1 - i), 1);
        return { key: month_key(d), label: month_label(d) };
    });
}

export function build_monthly_trend(bids: Bid[], n: number): MonthlyTrendPoint[] {
    const scaffold = build_month_scaffold(n);
    const counts = new Map<string, number>();

    bids.forEach((b) => {
        const d = to_date((b as any).endDate ?? (b as any).bidEnd ?? "");
        if (!d) return;
        const key = month_key(d);
        counts.set(key, (counts.get(key) ?? 0) + 1);
    });

    return scaffold.map((m) => ({ month: m.label, value: counts.get(m.key) ?? 0 }));
}

function normalize_region(raw: string): string {
    const v = (raw || "").trim();
    if (!v) return "기타";
    const keys = ["서울", "경기", "인천", "부산", "대구", "대전", "광주", "울산", "세종"];
    for (const k of keys) if (v.includes(k)) return k;
    return "기타";
}

export function build_region_dist(bids: Bid[]): RegionDistPoint[] {
    const map = new Map<string, number>();

    bids.forEach((b) => {
        const r = normalize_region((b as any).region ?? "");
        map.set(r, (map.get(r) ?? 0) + 1);
    });

    return Array.from(map.entries()).map(([name, value]) => ({ name, value }));
}
