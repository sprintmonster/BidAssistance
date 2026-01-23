import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";

export type RegionDistPoint = {
  name: string;
  value: number;
};

const COLORS = ["#3B82F6", "#8B5CF6", "#EC4899", "#10B981", "#F59E0B", "#06B6D4"];

function pct(value: number, total: number): number {
  if (total <= 0) return 0;
  return Math.round((value / total) * 1000) / 10;
}

export function RegionPieChart({
  data,
  loading,
}: {
  data: RegionDistPoint[];
  loading: boolean;
}) {
  const total = data.reduce((acc, cur) => acc + (cur.value || 0), 0);

  return (
    <div className="border rounded-2xl p-6 bg-white">
      <div className="mb-4">
        <div className="font-semibold">지역별 분포</div>
        <div className="text-sm text-gray-400">전체 공고 기준</div>
      </div>

      <div className="h-64">
        {loading ? (
          <div className="h-full w-full rounded-xl bg-gray-100 animate-pulse" />
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Tooltip
                formatter={(v: any, _name: any, props: any) => {
                  const value = Number(v) || 0;
                  const name = props?.payload?.name ?? "";
                  return [`${value}건 (${pct(value, total)}%)`, name];
                }}
              />
              <Legend
                formatter={(value: any, entry: any) => {
                  const v = Number(entry?.payload?.value) || 0;
                  return `${String(value)} (${v}건)`;
                }}
              />
              <Pie
                data={data}
                dataKey="value"
                nameKey="name"
                innerRadius={60}
                outerRadius={90}
                paddingAngle={3}
                labelLine={false}
                label={(d: any) => `${d.name} ${pct(d.value, total)}%`}
              >
                {data.map((_, idx) => (
                  <Cell key={`c-${idx}`} fill={COLORS[idx % COLORS.length]} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
