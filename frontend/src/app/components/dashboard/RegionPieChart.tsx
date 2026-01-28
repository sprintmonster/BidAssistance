import {
	Bar,
	BarChart,
	CartesianGrid,
	Cell,
	LabelList,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";

export type RegionDistPoint = {
	name: string;
	value: number;
};

const COLORS = [
	"#3B82F6",
	"#8B5CF6",
	"#EC4899",
	"#10B981",
	"#F59E0B",
	"#06B6D4",
];

function format_count(v: unknown): string {
	const n = Number(v);
	if (!Number.isFinite(n)) return "0건";
	return `${n.toLocaleString("ko-KR")}건`;
}

function normalize_data(data: RegionDistPoint[]): RegionDistPoint[] {
	return (Array.isArray(data) ? data : [])
		.map((d) => ({ name: String(d.name ?? ""), value: Number(d.value) || 0 }))
		.filter((d) => d.name && d.value > 0)
		.sort((a, b) => b.value - a.value);
}

export function RegionPieChart({
	data = [],
	loading = false,
}: {
	data?: RegionDistPoint[];
	loading?: boolean;
}) {
	const items = normalize_data(data);
	const maxNameLen = items.reduce((m, d) => Math.max(m, d.name.length), 0);
	const yWidth = Math.min(140, Math.max(72, maxNameLen * 10));

	return (
		<div className="border rounded-2xl p-6 bg-white">
			<div className="mb-4">
				<div className="font-semibold">지역별 분포</div>
				<div className="text-sm text-gray-400">전체 공고 기준</div>
			</div>

			<div className="h-64">
				{loading ? (
					<div className="h-full w-full rounded-xl bg-gray-100 animate-pulse" />
				) : items.length === 0 ? (
					<div className="h-full w-full rounded-xl bg-gray-50 flex items-center justify-center text-sm text-gray-400">
						데이터가 없습니다.
					</div>
				) : (
					<ResponsiveContainer width="100%" height="100%">
						<BarChart
							data={items}
							layout="vertical"
							margin={{ top: 8, right: 56, left: 8, bottom: 8 }}
						>
							<CartesianGrid strokeDasharray="3 3" vertical={false} />
							<XAxis type="number" tickLine={false} axisLine={false} />
							<YAxis
								type="category"
								dataKey="name"
								width={yWidth}
								tickLine={false}
								axisLine={false}
							/>
							<Tooltip formatter={(v: any) => format_count(v)} />
							<Bar dataKey="value" radius={[10, 10, 10, 10]}>
								<LabelList
									dataKey="value"
									position="right"
									formatter={(v: any) => format_count(v)}
								/>
								{items.map((_, idx) => (
									<Cell key={`c-${idx}`} fill={COLORS[idx % COLORS.length]} />
								))}
							</Bar>
						</BarChart>
					</ResponsiveContainer>
				)}
			</div>
		</div>
	);
}
