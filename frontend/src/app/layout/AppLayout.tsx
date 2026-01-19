import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { ChatbotFloatingButton } from "../components/ChatbotFloatingButton";
import { fetchWishlist } from "../api/wishlist";

export function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const isHome = location.pathname === "/";

  const [query, setQuery] = useState("");
  const [wishlistCount, setWishlistCount] = useState<number>(0);

  const isAuthed = useMemo(
    () => !!localStorage.getItem("accessToken"),
    [location.pathname]
  );

  useEffect(() => {
    if (!localStorage.getItem("accessToken")) {
      setWishlistCount(0);
      return;
    }
    fetchWishlist()
      .then((items) => setWishlistCount(items.length))
      .catch(() => setWishlistCount(0));
  }, [location.pathname]);

  const onSubmitSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    navigate(`/bids?q=${encodeURIComponent(q)}`);
  };

  const logout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("refreshToken");
    localStorage.removeItem("userId");
    localStorage.removeItem("name");
    localStorage.removeItem("email");
    navigate("/", { replace: true });
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* ===== Header (Top Bar) ===== */}
      <header className="border-b bg-slate-900 text-white">
        <div className="max-w-7xl mx-auto px-5 h-16 flex items-center justify-between gap-4">
          {/* Left: Logo */}
          <div className="flex items-center gap-3 min-w-[260px]">
            <button
              type="button"
              onClick={() => navigate("/")}
              aria-label="홈으로"
              className="h-16 flex items-center"
            >
              <span className="h-16 w-auto block object-contain">
                <img
                  src="/logo.png"
                  alt="입찰인사이트 로고"
                  className="h-16 w-auto block object-contain"
                />
              </span>
            </button>
          </div>

          {/* Center: MENU (홈에서는 숨김, 다른 페이지에서만 보임) */}
          {!isHome ? (
            <nav className="hidden md:flex items-center gap-2">
              <TopNavButton
                label="대시보드"
                active={location.pathname.startsWith("/dashboard")}
                onClick={() => navigate("/dashboard")}
              />
              <TopNavButton
                label="공고 찾기"
                active={location.pathname.startsWith("/bids")}
                onClick={() => navigate("/bids")}
              />
              <TopNavButton
                label="장바구니"
                active={location.pathname.startsWith("/cart")}
                badge={wishlistCount}
                onClick={() => navigate("/cart")}
              />
              <TopNavButton
                label="커뮤니티"
                active={location.pathname.startsWith("/community")}
                onClick={() => navigate("/community")}
              />
            </nav>
          ) : (
            <div className="flex-1" />
          )}

          {/* Right: Notice/Alarm (+ authed only) */}
          <div className="flex items-center gap-2 justify-end min-w-[260px]">
            <HeaderPill onClick={() => navigate("/notice")} label="공지사항" />
            <HeaderPill onClick={() => navigate("/notifications")} label="알림" />

            {isAuthed && (
              <>
                <HeaderPill onClick={() => navigate("/profile")} label="마이" />
                <button
                  onClick={logout}
                  className="ml-1 px-3 h-10 rounded-full bg-white/10 hover:bg-white/15 transition text-sm"
                >
                  로그아웃
                </button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* ===== Home Only: Search Bar Row (헤더 아래로 내림) ===== */}
      {isHome && (
        <section className="bg-white text-slate-900 border-b border-slate-200">
          <div className="max-w-7xl mx-auto px-5 py-6">
            <form
              onSubmit={onSubmitSearch}
              className="w-full max-w-[760px] mx-auto"
            >
              <div className="relative">
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="AI로 공고를 검색해보세요 (예: 서울/경기 10억~50억 시설공사, 마감 임박)"
                  className="w-full h-12 rounded-full bg-white text-slate-900 placeholder:text-slate-400 pl-5 pr-14
                             border border-blue-500/80 focus:border-blue-400 focus:ring-4 focus:ring-blue-500/20 outline-none"
                />
                <button
                  type="submit"
                  className="absolute right-1 top-1 h-10 w-12 rounded-full bg-blue-600 hover:bg-blue-500 transition flex items-center justify-center"
                  aria-label="검색"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                    <path
                      d="M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15Z"
                      stroke="white"
                      strokeWidth="2"
                    />
                    <path
                      d="M21 21l-4.35-4.35"
                      stroke="white"
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                  </svg>
                </button>
              </div>
            </form>
          </div>
        </section>
      )}

      {/* ===== Main ===== */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* ===== Footer ===== */}
      <footer className="border-t bg-white py-5 text-sm text-gray-500">
        <div className="max-w-7xl mx-auto px-5 flex items-center justify-between">
          <div>© 2026 입찰인사이트. All rights reserved.</div>
          <div className="flex gap-4">
            <button className="hover:text-gray-700">이용약관</button>
            <button className="hover:text-gray-700">개인정보처리방침</button>
            <button className="hover:text-gray-700">고객지원</button>
          </div>
        </div>
      </footer>

      <ChatbotFloatingButton />
    </div>
  );
}

function HeaderPill({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-4 h-10 rounded-full bg-white text-slate-900 border border-slate-200 hover:bg-slate-50 transition text-sm"
    >
      {label}
    </button>
  );
}

function TopNavButton({
  label,
  onClick,
  active,
  badge,
}: {
  label: string;
  onClick: () => void;
  active?: boolean;
  badge?: number;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "relative px-4 h-10 rounded-full text-sm transition",
        active
          ? "bg-blue-600 text-white"
          : "bg-white/10 hover:bg-white/15 text-white",
      ].join(" ")}
    >
      {label}
      {!!badge && badge > 0 && (
        <span className="absolute -right-1 -top-1 min-w-[18px] h-[18px] px-1 rounded-full bg-white text-slate-900 text-[11px] font-bold flex items-center justify-center">
          {badge}
        </span>
      )}
    </button>
  );
}
