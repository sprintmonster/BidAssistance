import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export type MonthlyTrendPoint = {
  month: string;
  value: number;
};

export function MonthlyTrendChart({
  data,
  loading,
}: {
  data: MonthlyTrendPoint[];
  loading: boolean;
}) {
  return (
    <div className="border rounded-2xl p-6 bg-white">
      <div className="mb-4">
        <div className="font-semibold">월별 공고 추이</div>
        <div className="text-sm text-gray-400">끝나는 날(endDate) 기준 · 앞으로 6개월</div>
      </div>

      <div className="h-64">
        {loading ? (
          <div className="h-full w-full rounded-xl bg-gray-100 animate-pulse" />
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis dataKey="month" />
              <YAxis allowDecimals={false} />
              <Tooltip formatter={(v) => [`${v}건`, "공고"]} />
              <Line type="monotone" dataKey="value" stroke="#2563EB" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
