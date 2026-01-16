import { Outlet, useNavigate, useLocation } from "react-router-dom";

export function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();

  // 홈페이지 여부
  const isHome = location.pathname === "/";

  return (
    <div className="min-h-screen flex flex-col bg-white">
      {/* ===== Header ===== */}
      <header className="border-b bg-white">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          {/* Logo */}
          <div
            className="flex items-center gap-3 cursor-pointer"
            onClick={() => navigate("/")}
          >
            <div className="w-8 h-8 bg-blue-600 rounded-lg" />
            <div>
              <div className="font-bold">입찰인사이트</div>
              <div className="text-xs text-gray-500">
                Smart Procurement Platform
              </div>
            </div>
          </div>

          {/* ===== 중앙 네비게이션 ===== */}
          {!isHome && (
            <nav className="flex gap-2">
              <NavButton label="대시보드" onClick={() => navigate("/dashboard")} />
              <NavButton label="공고 찾기" onClick={() => navigate("/bids")} />
              <NavButton label="장바구니" onClick={() => navigate("/cart")} />
              <NavButton label="커뮤니티" onClick={() => navigate("/community")} />
            </nav>
          )}

          {/* ===== 우측 버튼 ===== */}
          <div className="flex items-center gap-2">
            <IconButton label="공지사항" />
            <IconButton label="알림" />
          </div>
        </div>
      </header>

      {/* ===== Main ===== */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* ===== Footer ===== */}
      <footer className="border-t py-6 text-sm text-gray-500 text-center">
        © 2026 입찰인사이트. All rights reserved.
      </footer>
    </div>
  );
}

/* ===== UI Components ===== */

function NavButton({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="px-4 py-2 rounded-xl bg-blue-50 text-blue-700 text-sm hover:bg-blue-100 transition"
    >
      {label}
    </button>
  );
}

function IconButton({ label }: { label: string }) {
  return (
    <button className="px-3 py-2 rounded-lg border text-sm hover:bg-gray-50">
      {label}
    </button>
  );
}
