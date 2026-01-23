import {
	CartesianGrid,
	Line,
	LineChart,
	Pie,
	PieChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
	Legend,
	Cell,
} from "recharts";
import {useEffect, useState} from "react";
import {fetchBids} from "../api/bids";
import {fetchWishlist} from "../api/wishlist";

type Kpi = {
    newBidsToday: number;
    wishlistCount: number;
    closingSoon3Days: number;
    totalExpectedAmount: number; // 원 단위 합
};

const EMPTY_KPI: Kpi = {
    newBidsToday: 0,
    wishlistCount: 0,
    closingSoon3Days: 0,
    totalExpectedAmount: 0,
};

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

function addDays(d: Date, days: number): Date {
    const x = new Date(d);
    x.setDate(x.getDate() + days);
    return x;
}
function toNumberAmount(v: unknown): number {
    if (v == null || v === "") return 0;
    if (typeof v === "number") return Number.isFinite(v) ? v : 0;
    // "123,000" / "123000" / BigInt 문자열 모두 대응
    const n = Number(String(v).replace(/[^\d.-]/g, ""));
    return Number.isFinite(n) ? n : 0;
}

export function Dashboard() {
	// TODO: 추후 API 연동 시 여기 데이터만 교체하면 UI 유지됨
	const [kpi, setKpi] = useState<Kpi>(EMPTY_KPI);

    useEffect(() => {
        const load = async () => {
            const uidStr = localStorage.getItem("userId");
            const userId = Number(uidStr);

            const now = new Date();
            const todayStart = startOfToday(now);
            const todayEnd = endOfToday(now);
            const soonEnd = addDays(now, 3);

            const bids = await fetchBids();

            const newBidsToday = bids.filter((b) => {
                const s = parseDateSafe(b.startDate);
                return !!s && s >= todayStart && s <= todayEnd;
            }).length;

            const closingSoon3Days = bids.filter((b) => {
                const e = parseDateSafe(b.endDate);
                return !!e && e >= now && e <= soonEnd;
            }).length;

            // 로그인 안 했으면 wishlist는 0
            if (!uidStr || !Number.isFinite(userId)) {
                setKpi({
                    newBidsToday,
                    closingSoon3Days,
                    wishlistCount: 0,
                    totalExpectedAmount: 0,
                });
                return;
            }

            // 로그인 했을 때만 wishlist 불러오기
            const wishlist = await fetchWishlist(userId);

            setKpi({
                newBidsToday,
                closingSoon3Days,
                wishlistCount: wishlist.length,
                totalExpectedAmount: wishlist.reduce(
                    (sum, it) => sum + toNumberAmount(it.baseAmount),
                    0
                ),
            });
        };

        void load();
    }, []);



    const monthlyTrend = [
		{ month: "7월", value: 45 },
		{ month: "8월", value: 52 },
		{ month: "9월", value: 48 },
		{ month: "10월", value: 61 },
		{ month: "11월", value: 58 },
		{ month: "12월", value: 68 },
	];

	const regionDist = [
		{ name: "서울", value: 34 },
		{ name: "경기", value: 23 },
		{ name: "인천", value: 16 },
		{ name: "기타", value: 27 },
	];

	return (
		<div className="space-y-6">
			{/* KPI cards */}
			<div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
				<KpiCard
					title="신규 공고"
					value={`${kpi.newBidsToday}개`}
					sub="오늘 시작된 공고"
					icon="📄"
				/>
				<KpiCard
					title="관심 공고"
					value={`${kpi.wishlistCount}개`}
					sub="장바구니"
					icon="📈"
				/>
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

			{/* Charts */}
			<div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
				<div className="border rounded-2xl p-6 bg-white">
					<div className="mb-4">
						<div className="text-base font-semibold">월별 공고 추이</div>
						<div className="text-sm text-gray-500">최근 6개월</div>
					</div>

					<div className="h-[320px]">
						<ResponsiveContainer width="100%" height="100%">
							<LineChart data={monthlyTrend} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
								<CartesianGrid strokeDasharray="3 3" />
								<XAxis dataKey="month" />
								<YAxis />
								<Tooltip />
								<Line
									type="monotone"
									dataKey="value"
									stroke="#2563eb"
									strokeWidth={3}
									dot={{ r: 4 }}
								/>
							</LineChart>
						</ResponsiveContainer>
					</div>
				</div>

				<div className="border rounded-2xl p-6 bg-white">
					<div className="mb-4">
						<div className="text-base font-semibold">지역별 분포</div>
						<div className="text-sm text-gray-500">현재 진행 중인 공고</div>
					</div>

					<div className="h-[320px]">
						<ResponsiveContainer width="100%" height="100%">
							<PieChart>
								<Tooltip />
								<Legend />
								<Pie
									data={regionDist}
									dataKey="value"
									nameKey="name"
									outerRadius={110}
									label={(d) => `${d.name} ${d.value}%`}
								>
									{/* 원래 스샷 느낌대로 색 고정 */}
									{regionDist.map((_, idx) => (
										<Cell 
											key={`c-${idx}`}
											fill={["#3b82f6", "#8b5cf6", "#ec4899", "#10b981"][idx % 4]}
										/>
									))}
								</Pie>
							</PieChart>
						</ResponsiveContainer>
					</div>
				</div>
			</div>
		</div>
	);
}

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
