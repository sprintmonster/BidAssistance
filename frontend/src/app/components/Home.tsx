import { useNavigate } from "react-router-dom";

export function Home() {
  const navigate = useNavigate();

  return (
    <div className="space-y-8">
      {/* 상단 카드 4개 */}
      <div className="grid grid-cols-4 gap-4">
        <Card title="대시보드" onClick={() => navigate("/dashboard")} />
        <Card title="공고 찾기" onClick={() => navigate("/bids")} />
        <Card title="장바구니" onClick={() => navigate("/cart")} />
        <Card title="커뮤니티" onClick={() => navigate("/community")} />
      </div>

      {/* 메인 영역 */}
      <div className="grid grid-cols-3 gap-6">
        {/* AI 검색 */}
        <div className="col-span-2 border rounded-xl p-6 space-y-4">
          <h2 className="font-semibold">✨ AI 기반 공고 검색</h2>
          <p className="text-sm text-gray-600">
            자연어로 입력하면 조건을 해석해 공고 탐색/분석 흐름으로 연결합니다.
          </p>

          <input
            className="w-full h-11 rounded-md px-4 bg-gray-50"
            placeholder='예: "서울/경기 10억~50억 시설공사, 마감 임박 우선"'
          />

          <div className="flex gap-3">
            <button className="bg-black text-white px-4 py-2 rounded-md">
              AI로 검색
            </button>
            <button
              className="border px-4 py-2 rounded-md"
              onClick={() => navigate("/bids")}
            >
              공고 리스트로 이동
            </button>
          </div>
        </div>

        {/* 로그인 카드 */}
        <div className="border rounded-xl p-6 space-y-4">
          <h3 className="font-semibold text-lg">로그인</h3>
          <p className="text-sm text-gray-600">
            로그인하면 장바구니/알림/AI 기능을 이용할 수 있습니다.
          </p>

          <button
            className="w-full bg-black text-white py-2 rounded-md"
            onClick={() => navigate("/login")}
          >
            로그인 페이지로 이동
          </button>

          <button
            className="w-full border py-2 rounded-md"
            onClick={() => navigate("/register")}
          >
            회원가입
          </button>
        </div>
      </div>
    </div>
  );
}

function Card({
  title,
  onClick,
}: {
  title: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 border rounded-xl p-4 hover:bg-gray-50"
    >
      <div className="w-10 h-10 bg-gray-100 rounded-lg" />
      <span className="font-medium">{title}</span>
    </button>
  );
}
