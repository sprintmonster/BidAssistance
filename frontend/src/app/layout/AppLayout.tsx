import { Outlet, NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  useEffect,
  useMemo,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import { ChatbotFloatingButton } from "../components/ChatbotFloatingButton";
import {
  consume_reco_popup_trigger,
  is_reco_popup_suppressed_today,
  RecommendedBidsModal,
} from "../components/RecommendedBidsModal";
import { fetchWishlist } from "../api/wishlist";
import {
  LayoutDashboard,
  Search,
  ShoppingCart,
  Users,
  User,
  LogOut,
  Menu,
} from "lucide-react";
import { logout as apiLogout, checkLogin, persistLogin } from "../api/auth";

type SideItem = {
  label: string;
  to: string;
  icon: ReactNode;
  match: (path: string) => boolean;
  badge?: number;
};

export function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const isHome = location.pathname === "/";

  const [query, setQuery] = useState("");
  const [wishlistCount, setWishlistCount] = useState<number>(0);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [recoOpen, setRecoOpen] = useState(false);

  useEffect(() => {
    let ignore = false;

    checkLogin()
      .then((data) => {
        if (ignore) return;
        if (data) persistLogin(data);
      })
      .catch(() => {});

    return () => {
      ignore = true;
    };
  }, [location.pathname]);

  const isAuthed = useMemo(() => !!localStorage.getItem("userId"), [location.pathname]);

  // ✅ "로그인 직후 1회"만 팝업 오픈
  useEffect(() => {
    if (!localStorage.getItem("userId")) return;

    if (is_reco_popup_suppressed_today()) {
      // 트리거가 있더라도 오늘은 안 띄움 (한 번 소비 처리)
      consume_reco_popup_trigger();
      return;
    }

    // 세션 트리거가 있으면 1회만 오픈
    if (consume_reco_popup_trigger()) {
      setRecoOpen(true);
    }
  }, [location.pathname]);

  useEffect(() => {
    if (!localStorage.getItem("userId")) {
      setWishlistCount(0);
      return;
    }

    const raw = localStorage.getItem("userId");
    const userId = raw ? Number(raw) : NaN;

    if (!Number.isFinite(userId)) {
      setWishlistCount(0);
      return;
    }

    fetchWishlist(userId)
      .then((items) => setWishlistCount(items.length))
      .catch(() => setWishlistCount(0));
  }, [location.pathname]);

  const onSubmitSearch = (e: FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    navigate(`/bids?q=${encodeURIComponent(q)}`);
  };

  const onLogout = async () => {
    await apiLogout();
    navigate("/", { replace: true });
  };

  const sideItems: SideItem[] = [
    {
      label: "대시보드",
      to: "/dashboard",
      icon: <LayoutDashboard size={18} />,
      match: (p) => p.startsWith("/dashboard"),
    },
    {
      label: "공고 찾기",
      to: "/bids",
      icon: <Search size={18} />,
      match: (p) => p.startsWith("/bids"),
    },
    {
      label: "장바구니",
      to: "/cart",
      icon: <ShoppingCart size={18} />,
      match: (p) => p.startsWith("/cart"),
      badge: wishlistCount,
    },
    {
      label: "커뮤니티",
      to: "/community",
      icon: <Users size={18} />,
      match: (p) => p.startsWith("/community"),
    },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <header className="border-b bg-slate-900 text-white">
        <div className="h-16 w-full flex items-center">
          <div className="flex items-center gap-3 h-16 md:w-[260px] w-auto pl-5">
            {!isHome && (
              <button
                type="button"
                className="md:hidden h-10 w-10 rounded-md bg-white/10 hover:bg-white/15 transition flex items-center justify-center"
                aria-label="메뉴 열기"
                onClick={() => setMobileOpen(true)}
              >
                <Menu size={18} />
              </button>
            )}
            <button
              type="button"
              onClick={() => navigate("/")}
              aria-label="홈으로"
              className="h-16 flex items-center"
            >
              <img
                src="/logo.png"
                alt="입찰인사이트 로고"
                className="h-16 w-auto block object-contain"
              />
            </button>
          </div>

          <div className="flex-1 h-16 flex items-center justify-end gap-2 px-5">
            <HeaderPill onClick={() => navigate("/notice")} label="공지사항" />
            <HeaderPill onClick={() => navigate("/notifications")} label="알림" />
          </div>
        </div>
      </header>

      {!isHome && mobileOpen && (
        <div className="fixed inset-0 z-[9999]">
          <button
            type="button"
            className="absolute inset-0 bg-black/40"
            aria-label="메뉴 닫기"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="absolute left-0 top-0 h-full w-[280px] bg-slate-900 text-white shadow-xl">
            <SideBarContent
              items={sideItems}
              pathname={location.pathname}
              isAuthed={isAuthed}
              onLogout={onLogout}
              onNavigate={(to) => navigate(to)}
            />
          </aside>
        </div>
      )}

      {isHome && (
        <section className="bg-white text-slate-900 border-b border-slate-200">
          <div className="w-full px-5 py-6">
            <form onSubmit={onSubmitSearch} className="w-full max-w-[760px] mx-auto">
              <div className="relative">
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="공고를 키워드로 검색해보세요 (공고명/기관/예산/마감)"
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

      {isHome ? (
        <main className="flex-1">
          <Outlet />
        </main>
      ) : (
        <div className="flex-1 min-h-0 flex">
          <aside className="hidden md:flex w-[260px] shrink-0 bg-slate-900 text-white border-r border-white/10">
            <SideBarContent
              items={sideItems}
              pathname={location.pathname}
              isAuthed={isAuthed}
              onLogout={onLogout}
              onNavigate={(to) => navigate(to)}
            />
          </aside>

          <main className="flex-1 min-w-0 min-h-0 bg-slate-50">
            <div className="h-full min-h-0 pl-0 pr-6 py-6">
              <div className="w-full max-w-[1280px] pl-8">
                <Outlet />
              </div>
            </div>
          </main>
        </div>
      )}

      <footer className="border-t bg-white py-3 text-sm text-gray-500">
        <div className="w-full px-5 flex items-center justify-between">
          <div>© 2026 입찰인사이트. All rights reserved.</div>
          <div className="flex gap-4 text-xs">
            <button onClick={() => navigate("/terms")} className="hover:text-gray-500">
              이용약관
            </button>
            <button onClick={() => navigate("/privacy")} className="hover:text-gray-500">
              개인정보처리방침
            </button>
            <button onClick={() => navigate("/support")} className="hover:text-gray-500">
              고객지원
            </button>
          </div>
        </div>
      </footer>

      <ChatbotFloatingButton />
      <RecommendedBidsModal open={recoOpen} onOpenChange={setRecoOpen} />
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

function SideBarContent({
  items,
  pathname,
  isAuthed,
  onLogout,
  onNavigate,
}: {
  items: SideItem[];
  pathname: string;
  isAuthed: boolean;
  onLogout: () => void;
  onNavigate: (to: string) => void;
}) {
  return (
    <div className="h-full w-full flex flex-col">
      <div className="px-5 py-4">
        <div className="text-sm font-semibold tracking-wide text-slate-200">MENU</div>
      </div>

      <nav className="px-4 flex flex-col gap-1">
        {items.map((it) => (
          <SideNavLink
            key={it.to}
            to={it.to}
            label={it.label}
            icon={it.icon}
            active={it.match(pathname)}
            badge={it.badge}
          />
        ))}
      </nav>

      <div className="flex-1" />

      <div className="px-4 pb-4">
        <div className="border-t border-white/10 pt-3">
          {isAuthed ? (
            <div className="flex flex-col gap-1">
              <button
                type="button"
                onClick={() => onNavigate("/profile")}
                className="w-full h-11 px-3 rounded-md flex items-center gap-3 text-slate-100 hover:bg-white/5 transition"
              >
                <User size={18} />
                <span className="text-sm font-medium">마이페이지</span>
              </button>

              <button
                type="button"
                onClick={onLogout}
                className="w-full h-11 px-3 rounded-md flex items-center gap-3 text-slate-100 hover:bg-white/5 transition"
              >
                <LogOut size={18} />
                <span className="text-sm font-medium">로그아웃</span>
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => onNavigate("/login")}
              className="w-full h-11 px-3 rounded-md flex items-center gap-3 text-slate-100 hover:bg-white/5 transition"
            >
              <User size={18} />
              <span className="text-sm font-medium">로그인</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function SideNavLink({
  to,
  label,
  icon,
  active,
  badge,
}: {
  to: string;
  label: string;
  icon: ReactNode;
  active: boolean;
  badge?: number;
}) {
  return (
    <NavLink
      to={to}
      className={[
        "relative h-11 px-3 rounded-md",
        "flex items-center gap-3",
        "transition",
        active ? "bg-white/10 text-white" : "text-slate-200 hover:bg-white/5 hover:text-white",
      ].join(" ")}
    >
      <span className={active ? "text-white" : "text-slate-200"}>{icon}</span>
      <span className="text-sm font-medium">{label}</span>

      {!!badge && badge > 0 && (
        <span className="ml-auto min-w-[20px] h-[20px] px-1 rounded-full bg-blue-600 text-white text-[12px] font-bold flex items-center justify-center">
          {badge}
        </span>
      )}
    </NavLink>
  );
}
