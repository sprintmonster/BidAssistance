import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const data = [
  { month: "7월", value: 45 },
  { month: "8월", value: 52 },
  { month: "9월", value: 48 },
  { month: "10월", value: 61 },
  { month: "11월", value: 58 },
  { month: "12월", value: 67 },
];

export function MonthlyTrendChart() {
  return (
    <div className="border rounded-2xl p-6">
      <div className="mb-4">
        <div className="font-semibold">월별 공고 추이</div>
        <div className="text-sm text-gray-400">
          최근 6개월
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#E5E7EB"
            />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#2563EB"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
