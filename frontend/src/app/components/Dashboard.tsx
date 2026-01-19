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
} from "recharts";

export function Dashboard() {
	// TODO: 추후 API 연동 시 여기 데이터만 교체하면 UI 유지됨
	const kpi = {
		newBidsThisMonth: 67,
		wishlistCount: 0,
		closingSoon3Days: 8,
		totalExpectedAmountEok: 142, // "억" 단위
	};

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
					value={`${kpi.newBidsThisMonth}개`}
					sub="이번 달"
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
					value={`${kpi.totalExpectedAmountEok}억`}
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
										<cell
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
