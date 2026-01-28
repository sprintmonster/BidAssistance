import { useMemo, useState } from "react";
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
	"#22C55E",
	"#A855F7",
	"#EF4444",
	"#0EA5E9",
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

function clamp(n: number, min: number, max: number): number {
	if (n < min) return min;
	if (n > max) return max;
	return n;
}

function slice_page<T>(items: T[], page: number, size: number): T[] {
	const start = page * size;
	return items.slice(start, start + size);
}

export function RegionPieChart({
	data = [],
	loading = false,
}: {
	data?: RegionDistPoint[];
	loading?: boolean;
}) {
	const PAGE_SIZE = 6;

	const items = useMemo(() => normalize_data(data), [data]);
	const pageCount = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
	const [page, setPage] = useState(0);

	const safePage = clamp(page, 0, pageCount - 1);
	const pageItems = useMemo(
		() => slice_page(items, safePage, PAGE_SIZE),
		[items, safePage],
	);

	const canPrev = safePage > 0;
	const canNext = safePage < pageCount - 1;

	const pageBaseIndex = safePage * PAGE_SIZE;

	return (
		<div className="border rounded-2xl p-6 bg-white">
			<div className="mb-4 flex items-start justify-between gap-3">
				<div>
					<div className="font-semibold">지역별 분포</div>
					<div className="text-sm text-gray-400">전체 공고 기준</div>
				</div>

				<div className="flex items-center gap-2">
					<div className="text-xs text-gray-400 tabular-nums">
						{items.length === 0 ? "0/0" : `${safePage + 1}/${pageCount}`}
					</div>
					<button
						type="button"
						className={[
							"h-9 w-9 rounded-full border bg-white inline-flex items-center justify-center",
							canPrev ? "hover:bg-gray-50" : "opacity-40 cursor-not-allowed",
						].join(" ")}
						disabled={!canPrev}
						onClick={() => setPage((p) => clamp(p - 1, 0, pageCount - 1))}
						aria-label="이전"
						title="이전"
					>
						<span className="text-lg leading-none">‹</span>
					</button>
					<button
						type="button"
						className={[
							"h-9 w-9 rounded-full border bg-white inline-flex items-center justify-center",
							canNext ? "hover:bg-gray-50" : "opacity-40 cursor-not-allowed",
						].join(" ")}
						disabled={!canNext}
						onClick={() => setPage((p) => clamp(p + 1, 0, pageCount - 1))}
						aria-label="다음"
						title="다음"
					>
						<span className="text-lg leading-none">›</span>
					</button>
				</div>
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
						<BarChart data={pageItems} margin={{ top: 16, right: 16, left: 8, bottom: 44 }}>
							<CartesianGrid strokeDasharray="3 3" vertical={false} />
							<XAxis
								dataKey="name"
								tickLine={false}
								axisLine={false}
								interval={0}
								angle={-18}
								textAnchor="end"
								height={54}
							/>
							<YAxis type="number" tickLine={false} axisLine={false} />
							<Tooltip formatter={(v: any) => format_count(v)} />
							<Bar dataKey="value" radius={[10, 10, 0, 0]}>
								<LabelList
									dataKey="value"
									position="top"
									formatter={(v: any) => format_count(v)}
								/>
								{pageItems.map((_, idx) => (
									<Cell
										key={`c-${idx}`}
										fill={COLORS[(pageBaseIndex + idx) % COLORS.length]}
									/>
								))}
							</Bar>
						</BarChart>
					</ResponsiveContainer>
				)}
			</div>
		</div>
	);
}
