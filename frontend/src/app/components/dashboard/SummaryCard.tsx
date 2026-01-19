import {
  FileText,
  Heart,
  AlertCircle,
  DollarSign,
} from "lucide-react";

const items = [
  {
    title: "신규 공고",
    value: "67개",
    sub: "이번 달",
    icon: FileText,
  },
  {
    title: "관심 공고",
    value: "0개",
    sub: "장바구니",
    icon: Heart,
  },
  {
    title: "마감 임박",
    value: "8개",
    sub: "3일 이내",
    icon: AlertCircle,
    highlight: true,
  },
  {
    title: "총 예상액",
    value: "142억",
    sub: "관심 공고 합계",
    icon: DollarSign,
  },
];

export function SummaryCards() {
  return (
    <div className="grid grid-cols-4 gap-6">
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <div
            key={item.title}
            className="border rounded-2xl p-6 flex justify-between"
          >
            <div className="space-y-2">
              <div className="text-sm text-gray-500">
                {item.title}
              </div>
              <div className="text-2xl font-semibold text-gray-900">
                {item.value}
              </div>
              <div className="text-xs text-gray-400">
                {item.sub}
              </div>
            </div>

            <div
              className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                item.highlight
                  ? "bg-orange-100 text-orange-500"
                  : "bg-gray-100 text-gray-400"
              }`}
            >
              <Icon size={18} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

