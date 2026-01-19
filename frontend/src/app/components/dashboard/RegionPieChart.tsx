import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
} from "recharts";

const data = [
  { name: "서울", value: 34, color: "#3B82F6" },
  { name: "경기", value: 23, color: "#8B5CF6" },
  { name: "인천", value: 16, color: "#EC4899" },
  { name: "기타", value: 27, color: "#10B981" },
];

export function RegionPieChart() {
  return (
    <div className="border rounded-2xl p-6">
      <div className="mb-4">
        <div className="font-semibold">지역별 분포</div>
        <div className="text-sm text-gray-400">
          현재 진행 중인 공고
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              innerRadius={60}
              outerRadius={90}
              paddingAngle={3}
            >
              {data.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={entry.color}
                />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
