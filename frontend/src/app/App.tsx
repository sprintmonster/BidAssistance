import { useEffect, useMemo, useState } from "react";
import { LoginPage } from "./components/LoginPage";
import { SignupPage } from "./components/SignupPage";
import { Dashboard } from "./components/Dashboard";
import { BidDiscovery } from "./components/BidDiscovery";
import { BidSummary } from "./components/BidSummary";
import { CartPage } from "./components/CartPage";
import { NotificationsPage, type NotificationItem } from "./components/NotificationsPage";
import { NoticePage } from "./components/NoticePage";
import { ChatbotPage } from "./components/ChatbotPage";
import { ProfilePage } from "./components/ProfilePage";

import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Input } from "./components/ui/input";

import type { Page } from "../types/navigation";

import {
  LayoutDashboard,
  Search,
  ShoppingCart,
  Bell,
  User,
  Building2,
  LogOut,
  Menu,
  X,
  Megaphone,
  MessageSquare,
  Sparkles,
} from "lucide-react";

import { toast, Toaster } from "sonner";

/**
 * history state
 */
type NavState = { page: Page; bidId?: number };

function isNavState(v: unknown): v is NavState {
  if (!v || typeof v !== "object") return false;
  const anyV = v as Record<string, unknown>;
  return typeof anyV.page === "string";
}

function isPublicPage(page: Page) {
  return page === "home" || page === "login" || page === "signup" || page === "notice";
}

/** 추천공고 타입/목록(데모용: 실제 API/추천엔진으로 교체 가능) */
type RecommendedBid = {
  id: number;
  title: string;
  agency: string;
  region: string;
  budget: string;
  deadline: string;
  tags: string[];
};

const DEFAULT_RECOMMENDED_BIDS: RecommendedBid[] = [
  {
    id: 2,
    title: "경기도 성남시 공공건물 신축공사",
    agency: "경기도 성남시청",
    region: "경기",
    budget: "87억 원",
    deadline: "2026-01-15",
    tags: ["대형", "신축", "경기권"],
  },
  {
    id: 1,
    title: "서울시 강남구 도로 보수공사",
    agency: "서울특별시 강남구청",
    region: "서울",
    budget: "35억 원",
    deadline: "2026-01-08",
    tags: ["마감임박", "도로", "서울"],
  },
  {
    id: 6,
    title: "광주시 광산구 문화체육시설 신축",
    agency: "광주광역시 광산구청",
    region: "광주",
    budget: "45억 원",
    deadline: "2026-01-18",
    tags: ["신축", "공공", "광주"],
  },
  {
    id: 4,
    title: "부산시 해운대구 주차장 건설",
    agency: "부산광역시 해운대구청",
    region: "부산",
    budget: "23억 원",
    deadline: "2026-01-12",
    tags: ["주차장", "부산", "중형"],
  },
];

function formatDday(deadline: string) {
  const today = new Date();
  const d = new Date(deadline);
  const diff = d.getTime() - today.getTime();
  const days = Math.ceil(diff / (1000 * 60 * 60 * 24));
  if (Number.isNaN(days)) return "";
  if (days < 0) return "마감";
  if (days === 0) return "D-Day";
  return `D-${days}`;
}

/**
 * 추천공고 팝업 (1회성)
 * - 닫기(ESC/바깥클릭 포함) 하면 영구적으로 다시 안 뜸
 */
function RecommendedBidsModal({
  open,
  onClose,
  bids,
  onView,
  onAddToCart,
}: {
  open: boolean;
  onClose: () => void;
  bids: RecommendedBid[];
  onView: (bidId: number) => void;
  onAddToCart: (bidId: number) => void;
}) {
  useEffect(() => {
    if (!open) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/40 flex items-end sm:items-center justify-center p-3"
      onMouseDown={onClose}
    >
      <div
        className="bg-white w-full sm:max-w-3xl h-[85vh] rounded-xl shadow-xl flex flex-col overflow-hidden"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="px-4 py-3 border-b flex items-center justify-between">
          <div className="font-semibold">추천 공고</div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex-1 overflow-auto p-4 space-y-3">
          <Card className="bg-sky-50 border-sky-200">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">맞춤 추천 공고</CardTitle>
              <CardDescription className="text-sm">
                관심 지역/금액/최근 조회 흐름을 기반으로 추천됩니다. (현재는 데모 데이터)
              </CardDescription>
            </CardHeader>
          </Card>

          <div className="space-y-3">
            {bids.map((b) => (
              <Card key={b.id} className="border">
                <CardContent className="p-4 space-y-2">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="font-semibold truncate">{b.title}</div>
                      <div className="text-sm text-muted-foreground">{b.agency}</div>
                    </div>

                    <div className="shrink-0 flex items-center gap-2">
                      <Badge variant="outline">{b.region}</Badge>
                      <Badge className="bg-sky-600">{formatDday(b.deadline)}</Badge>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary">{b.budget}</Badge>
                    <Badge variant="secondary">마감: {b.deadline}</Badge>
                    {b.tags.map((t) => (
                      <Badge key={t} variant="outline">
                        {t}
                      </Badge>
                    ))}
                  </div>

                  <div className="flex gap-2 pt-1">
                    <Button size="sm" onClick={() => onView(b.id)}>
                      상세 보기
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => onAddToCart(b.id)} className="gap-2">
                      <ShoppingCart className="h-4 w-4" />
                      장바구니 담기
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        <div className="border-t px-4 py-3 flex justify-end">
          <Button variant="outline" onClick={onClose}>
            닫기
          </Button>
        </div>
      </div>
    </div>
  );
}

/** 챗봇 모달 */
function ChatbotModal({
  open,
  onClose,
  isAuthenticated,
  onGoLogin,
}: {
  open: boolean;
  onClose: () => void;
  isAuthenticated: boolean;
  onGoLogin: () => void;
}) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/40 flex items-end sm:items-center justify-center p-3"
      onMouseDown={onClose}
    >
      <div
        className="bg-white w-full sm:max-w-3xl h-[82vh] rounded-xl shadow-xl flex flex-col overflow-hidden"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="px-4 py-3 border-b flex items-center justify-between">
          <div className="font-semibold">AI 챗봇</div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex-1 overflow-auto">
          {isAuthenticated ? (
            <div className="p-4">
              <ChatbotPage />
            </div>
          ) : (
            <div className="p-6 space-y-3">
              <div className="font-semibold">로그인이 필요합니다</div>
              <div className="text-sm text-muted-foreground">
                챗봇 기능을 사용하려면 로그인 후 이용해 주세요.
              </div>
              <div className="flex gap-2">
                <Button
                  onClick={() => {
                    onClose();
                    onGoLogin();
                  }}
                >
                  로그인으로 이동
                </Button>
                <Button variant="outline" onClick={onClose}>
                  닫기
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** Home 본문(헤더/푸터 제외) */
function HomeContent({
  isAuthenticated,
  userEmail,
  cartCount,
  onNavigate,
  onGoLogin,
  onGoSignup,
  onLogout,
  onOpenChatbot,
}: {
  isAuthenticated: boolean;
  userEmail: string;
  cartCount: number;
  onNavigate: (page: Page) => void;
  onGoLogin: () => void;
  onGoSignup: () => void;
  onLogout: () => void;
  onOpenChatbot: () => void;
}) {
  const [query, setQuery] = useState("");

  const quickLinks = useMemo(
    () => [
      { id: "dashboard" as Page, label: "대시보드", icon: LayoutDashboard },
      { id: "bids" as Page, label: "공고 찾기", icon: Search },
      { id: "cart" as Page, label: "장바구니", icon: ShoppingCart, badge: cartCount },
      { id: "profile" as Page, label: "마이페이지", icon: User },
    ],
    [cartCount]
  );

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-9 space-y-6">
        {/* 퀵 링크 박스들 (파스텔톤 하늘색) */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {quickLinks.map((x) => {
            const Icon = x.icon;
            return (
              <button key={x.id} onClick={() => onNavigate(x.id)} className="text-left">
                <Card className="bg-sky-50 border-sky-200 hover:bg-sky-100/60 transition">
                  <CardContent className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="h-9 w-9 rounded-md bg-sky-100 flex items-center justify-center">
                        <Icon className="h-4 w-4" />
                      </div>
                      <div className="font-medium">{x.label}</div>
                    </div>
                    {"badge" in x && x.badge && x.badge > 0 ? <Badge>{x.badge}</Badge> : null}
                  </CardContent>
                </Card>
              </button>
            );
          })}
        </div>

        {/* AI 검색 패널 */}
        <Card className="bg-sky-50 border-sky-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5" />
              AI를 활용한 검색
            </CardTitle>
            <CardDescription>
              자연어로 검색 조건을 입력하면, 공고 탐색/분석 흐름으로 연결합니다.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder='예: "서울/경기 10억~50억 시설공사, 마감 임박 우선"'
              className="bg-white"
            />

            <div className="flex gap-2">
              <Button
                className="gap-2"
                onClick={() => {
                  if (!query.trim()) {
                    toast.info("검색어를 입력해 주세요.");
                    return;
                  }
                  localStorage.setItem("chatbot.initialQuery", query.trim());
                  onOpenChatbot();
                }}
              >
                <Sparkles className="h-4 w-4" />
                AI로 검색
              </Button>

              <Button variant="outline" onClick={() => onNavigate("bids")}>
                공고 리스트로 이동
              </Button>
            </div>

            <p className="text-sm text-muted-foreground">
              로그인 후 검색 결과 탐색, 저장, 알림 연동 기능을 이용할 수 있습니다.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* 우측(더 얇게) */}
      <div className="col-span-12 lg:col-span-3">
        <div className="lg:max-w-xs lg:ml-auto">
          {!isAuthenticated ? (
            <Card className="bg-sky-50 border-sky-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">로그인</CardTitle>
                <CardDescription className="text-sm">
                  로그인하면 공고/장바구니/알림/AI 기능을 이용할 수 있습니다.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button size="sm" className="w-full" onClick={onGoLogin}>
                  로그인 페이지로 이동
                </Button>
                <Button size="sm" variant="outline" className="w-full" onClick={onGoSignup}>
                  회원가입
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Card className="bg-sky-50 border-sky-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">회원 정보</CardTitle>
                <CardDescription className="text-sm">현재 로그인된 계정 정보입니다.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <div className="font-semibold">김철수</div>
                  <div className="text-sm text-muted-foreground">{userEmail}</div>
                  <div className="flex gap-2 pt-2">
                    <Badge>호반건설</Badge>
                    <Badge variant="outline">중형 건설사</Badge>
                  </div>
                </div>

                <Button size="sm" variant="outline" className="w-full" onClick={onLogout}>
                  로그아웃
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("home");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userEmail, setUserEmail] = useState("");
  const [cartItems, setCartItems] = useState<number[]>([]);
  const [selectedBidId, setSelectedBidId] = useState<number | undefined>();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // 알림
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const unreadCount = useMemo(() => notifications.filter((n) => !n.read).length, [notifications]);

  const handleMarkRead = (id: number) => {
    setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, read: true } : n)));
  };

  const handleMarkAllRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  // 챗봇
  const [chatbotOpen, setChatbotOpen] = useState(false);

  // 추천공고 (1회성 팝업)
  const [recommendedBids] = useState<RecommendedBid[]>(DEFAULT_RECOMMENDED_BIDS);
  const [recommendationsOpen, setRecommendationsOpen] = useState(false);
  const RECO_DISMISSED_KEY = "bidassistance_reco_dismissed_v1";

  const navigateTo = (page: Page, bidId?: number, replace: boolean = false) => {
    const next: NavState = { page, bidId };
    if (replace) window.history.replaceState(next, "");
    else window.history.pushState(next, "");
    setCurrentPage(page);
    setSelectedBidId(bidId);
    setMobileMenuOpen(false);
  };

  // 추천공고 닫기: 닫는 순간 영구적으로 다시 안 뜨게 저장
  const closeRecommendationsPermanently = () => {
    localStorage.setItem(RECO_DISMISSED_KEY, "1");
    setRecommendationsOpen(false);
  };

  const handleViewRecommended = (bidId: number) => {
    // 상세보기로 이동하는 것도 "팝업 종료"에 해당하므로 영구 닫힘 처리
    closeRecommendationsPermanently();
    navigateTo("summary", bidId);
  };

  const handleAddToCart = (bidId: number) => {
    if (!cartItems.includes(bidId)) {
      setCartItems([...cartItems, bidId]);
      toast.success("장바구니에 추가되었습니다");
    } else {
      toast.info("이미 장바구니에 있는 공고입니다");
    }
  };

  const handleRemoveFromCart = (bidId: number) => {
    setCartItems(cartItems.filter((id) => id !== bidId));
    toast.success("장바구니에서 제거되었습니다");
  };

  // ---- history sync ----
  useEffect(() => {
    const st = window.history.state;

    if (isNavState(st) && st.page) {
      if (!isAuthenticated && !isPublicPage(st.page)) {
        toast.info("로그인이 필요합니다.");
        window.history.replaceState({ page: "login" } satisfies NavState, "");
        setCurrentPage("login");
        setSelectedBidId(undefined);
      } else {
        setCurrentPage(st.page as Page);
        setSelectedBidId(st.bidId as number | undefined);
      }
    } else {
      window.history.replaceState({ page: "home" } satisfies NavState, "");
      setCurrentPage("home");
      setSelectedBidId(undefined);
    }

    const onPopState = (e: PopStateEvent) => {
      if (!isNavState(e.state) || !e.state.page) return;

      const nextPage = e.state.page;

      if (!isAuthenticated && !isPublicPage(nextPage)) {
        toast.info("로그인이 필요합니다.");
        window.history.replaceState({ page: "login" } satisfies NavState, "");
        setCurrentPage("login");
        setSelectedBidId(undefined);
        setMobileMenuOpen(false);
        return;
      }

      setCurrentPage(nextPage);
      setSelectedBidId(e.state.bidId);
      setMobileMenuOpen(false);
    };

    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [isAuthenticated]);

  // 로그인/회원가입
  const handleLogin = (email: string) => {
    setIsAuthenticated(true);
    setUserEmail(email);
    navigateTo("home", undefined, true);
    toast.success("로그인되었습니다");

    // 로그인 직후: 추천공고 팝업 1회 자동 노출 (이미 닫은 적 있으면 절대 안 뜸)
    const dismissed = localStorage.getItem(RECO_DISMISSED_KEY) === "1";
    if (!dismissed) setRecommendationsOpen(true);
  };

  const handleSignup = (email: string) => {
    setIsAuthenticated(true);
    setUserEmail(email);
    navigateTo("home", undefined, true);
    toast.success("회원가입이 완료되었습니다");

    // 회원가입 직후도 로그인과 동일하게 1회 노출
    const dismissed = localStorage.getItem(RECO_DISMISSED_KEY) === "1";
    if (!dismissed) setRecommendationsOpen(true);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserEmail("");
    setCartItems([]);
    setSelectedBidId(undefined);
    setNotifications([]);
    setRecommendationsOpen(false);
    navigateTo("home", undefined, true);
    toast.info("로그아웃되었습니다");
  };

  const handleNavigate = (page: Page, bidId?: number) => {
    if (page === "chatbot") {
      setChatbotOpen(true);
      return;
    }

    if (!isAuthenticated && !isPublicPage(page)) {
      toast.info("로그인이 필요합니다.");
      navigateTo("login", undefined, true);
      return;
    }

    navigateTo(page, bidId, false);
  };

  // 헤더 중앙(다른 페이지에서만): 4개 버튼
  const headerQuickNav = useMemo(
    () => [
      { id: "dashboard" as Page, label: "대시보드", icon: LayoutDashboard },
      { id: "bids" as Page, label: "공고 찾기", icon: Search },
      { id: "cart" as Page, label: "장바구니", icon: ShoppingCart, badge: cartItems.length },
      { id: "profile" as Page, label: "마이페이지", icon: User },
    ],
    [cartItems.length]
  );

  // -------------------------
  // 비로그인
  // -------------------------
  if (!isAuthenticated) {
    if (currentPage === "signup") {
      return (
        <>
          <Toaster position="top-right" />
          <SignupPage onSignup={handleSignup} onNavigateToLogin={() => navigateTo("login")} />
        </>
      );
    }

    if (currentPage === "login") {
      return (
        <>
          <Toaster position="top-right" />
          <LoginPage onLogin={handleLogin} onNavigateToSignup={() => navigateTo("signup")} />
        </>
      );
    }

    return (
      <div className="min-h-screen bg-gray-50">
        <Toaster position="top-right" />

        <header className="bg-white border-b sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <button
                className="flex items-center gap-3 text-left"
                onClick={() => navigateTo("home", undefined, true)}
                type="button"
              >
                <div className="bg-blue-600 p-2 rounded-lg">
                  <Building2 className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold">입찰 인텔리전스</h1>
                  <p className="text-xs text-muted-foreground hidden sm:block">
                    Smart Procurement Platform
                  </p>
                </div>
              </button>

              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" className="gap-2" onClick={() => handleNavigate("notice")}>
                  <Megaphone className="h-4 w-4" />
                  공지사항
                </Button>

                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  onClick={() => handleNavigate("notifications")}
                >
                  <Bell className="h-4 w-4" />
                  알림
                  {unreadCount > 0 ? (
                    <Badge className="ml-2 h-5 w-5 flex items-center justify-center p-0 text-xs">
                      {unreadCount}
                    </Badge>
                  ) : null}
                </Button>
              </div>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {currentPage === "home" && (
            <HomeContent
              isAuthenticated={false}
              userEmail=""
              cartCount={0}
              onNavigate={(p) => handleNavigate(p)}
              onGoLogin={() => navigateTo("login")}
              onGoSignup={() => navigateTo("signup")}
              onLogout={() => undefined}
              onOpenChatbot={() => setChatbotOpen(true)}
            />
          )}

          {currentPage === "notice" && <NoticePage onNavigate={handleNavigate} canWrite={false} />}
        </main>

        <footer className="bg-white border-t mt-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <p className="text-sm text-muted-foreground">© 2026 입찰 인텔리전스. All rights reserved.</p>
              <div className="flex gap-4 text-sm text-muted-foreground">
                <a href="#" className="hover:text-blue-600">
                  이용약관
                </a>
                <a href="#" className="hover:text-blue-600">
                  개인정보처리방침
                </a>
                <a href="#" className="hover:text-blue-600">
                  고객지원
                </a>
              </div>
            </div>
          </div>
        </footer>

        <button
          type="button"
          onClick={() => setChatbotOpen(true)}
          className="fixed bottom-6 right-6 z-40"
          aria-label="AI 챗봇 열기"
        >
          <div className="shadow-lg rounded-full">
            <Button className="rounded-full h-12 w-12 p-0">
              <MessageSquare className="h-5 w-5" />
            </Button>
          </div>
        </button>

        <ChatbotModal
          open={chatbotOpen}
          onClose={() => setChatbotOpen(false)}
          isAuthenticated={false}
          onGoLogin={() => navigateTo("login", undefined, true)}
        />
      </div>
    );
  }

  // -------------------------
  // 로그인 후
  // -------------------------
  const showHeaderQuickNav = currentPage !== "home";

  return (
    <div className="min-h-screen bg-gray-50">
      <Toaster position="top-right" />

      <header className="bg-white border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 gap-3">
            <button
              className="flex items-center gap-3 text-left shrink-0"
              onClick={() => handleNavigate("home")}
              type="button"
            >
              <div className="bg-blue-600 p-2 rounded-lg">
                <Building2 className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold">입찰 인텔리전스</h1>
                <p className="text-xs text-muted-foreground hidden sm:block">
                  Smart Procurement Platform
                </p>
              </div>
            </button>

            {showHeaderQuickNav && (
              <nav className="hidden lg:flex flex-1 justify-center gap-1">
                {headerQuickNav.map((item) => {
                  const Icon = item.icon;
                  const isActive = currentPage === item.id;
                  return (
                    <Button
                      key={item.id}
                      variant={isActive ? "default" : "ghost"}
                      size="sm"
                      onClick={() => handleNavigate(item.id)}
                      className="relative"
                    >
                      <Icon className="h-4 w-4 mr-2" />
                      {item.label}
                      {"badge" in item && item.badge && item.badge > 0 ? (
                        <Badge className="ml-2 h-5 w-5 flex items-center justify-center p-0 text-xs">
                          {item.badge}
                        </Badge>
                      ) : null}
                    </Button>
                  );
                })}
              </nav>
            )}

            <div className="flex items-center gap-2 shrink-0">
              <Button
                variant="outline"
                size="sm"
                className="gap-2 hidden sm:flex"
                onClick={() => handleNavigate("notice")}
              >
                <Megaphone className="h-4 w-4" />
                공지사항
              </Button>

              <Button
                variant="outline"
                size="sm"
                className="gap-2 hidden sm:flex"
                onClick={() => handleNavigate("notifications")}
              >
                <Bell className="h-4 w-4" />
                알림
                {unreadCount > 0 ? (
                  <Badge className="ml-2 h-5 w-5 flex items-center justify-center p-0 text-xs">
                    {unreadCount}
                  </Badge>
                ) : null}
              </Button>

              <Button variant="ghost" size="sm" onClick={handleLogout} className="hidden lg:flex">
                <LogOut className="h-4 w-4 mr-2" />
                로그아웃
              </Button>

              <Button
                variant="ghost"
                size="sm"
                className="lg:hidden"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              >
                {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </div>

        {mobileMenuOpen && (
          <div className="lg:hidden border-t bg-white">
            <nav className="px-4 py-2 space-y-1">
              <Button
                variant={currentPage === "dashboard" ? "default" : "ghost"}
                className="w-full justify-start"
                onClick={() => handleNavigate("dashboard")}
              >
                <LayoutDashboard className="h-4 w-4 mr-3" />
                대시보드
              </Button>

              <Button
                variant={currentPage === "bids" ? "default" : "ghost"}
                className="w-full justify-start"
                onClick={() => handleNavigate("bids")}
              >
                <Search className="h-4 w-4 mr-3" />
                공고 찾기
              </Button>

              <Button
                variant={currentPage === "cart" ? "default" : "ghost"}
                className="w-full justify-start"
                onClick={() => handleNavigate("cart")}
              >
                <ShoppingCart className="h-4 w-4 mr-3" />
                장바구니
                {cartItems.length > 0 ? (
                  <Badge className="ml-auto h-5 w-5 flex items-center justify-center p-0 text-xs">
                    {cartItems.length}
                  </Badge>
                ) : null}
              </Button>

              <Button
                variant={currentPage === "profile" ? "default" : "ghost"}
                className="w-full justify-start"
                onClick={() => handleNavigate("profile")}
              >
                <User className="h-4 w-4 mr-3" />
                마이페이지
              </Button>

              <Button
                variant={currentPage === "notice" ? "default" : "ghost"}
                className="w-full justify-start"
                onClick={() => handleNavigate("notice")}
              >
                <Megaphone className="h-4 w-4 mr-3" />
                공지사항
              </Button>

              <Button
                variant={currentPage === "notifications" ? "default" : "ghost"}
                className="w-full justify-start"
                onClick={() => handleNavigate("notifications")}
              >
                <Bell className="h-4 w-4 mr-3" />
                알림
                {unreadCount > 0 ? (
                  <Badge className="ml-auto h-5 w-5 flex items-center justify-center p-0 text-xs">
                    {unreadCount}
                  </Badge>
                ) : null}
              </Button>

              <Button variant="ghost" className="w-full justify-start" onClick={handleLogout}>
                <LogOut className="h-4 w-4 mr-3" />
                로그아웃
              </Button>
            </nav>
          </div>
        )}
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentPage === "home" && (
          <HomeContent
            isAuthenticated
            userEmail={userEmail}
            cartCount={cartItems.length}
            onNavigate={(p) => handleNavigate(p)}
            onGoLogin={() => navigateTo("login")}
            onGoSignup={() => navigateTo("signup")}
            onLogout={handleLogout}
            onOpenChatbot={() => setChatbotOpen(true)}
          />
        )}

        {currentPage === "notice" && <NoticePage onNavigate={handleNavigate} canWrite={true} />}

        {currentPage === "dashboard" && <Dashboard onNavigate={handleNavigate} cart={cartItems} />}

        {currentPage === "bids" && (
          <BidDiscovery onNavigate={handleNavigate} onAddToCart={handleAddToCart} />
        )}

        {currentPage === "summary" && <BidSummary bidId={selectedBidId} onNavigate={handleNavigate} />}

        {currentPage === "cart" && (
          <CartPage
            cartItems={cartItems}
            onRemoveFromCart={handleRemoveFromCart}
            onNavigate={handleNavigate}
          />
        )}

        {currentPage === "notifications" && (
          <NotificationsPage
            onNavigate={handleNavigate}
            notifications={notifications}
            onMarkRead={handleMarkRead}
            onMarkAllRead={handleMarkAllRead}
          />
        )}

        {currentPage === "profile" && <ProfilePage userEmail={userEmail} />}
      </main>

      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-sm text-muted-foreground">© 2026 입찰 인텔리전스. All rights reserved.</p>
            <div className="flex gap-4 text-sm text-muted-foreground">
              <a href="#" className="hover:text-blue-600">
                이용약관
              </a>
              <a href="#" className="hover:text-blue-600">
                개인정보처리방침
              </a>
              <a href="#" className="hover:text-blue-600">
                고객지원
              </a>
            </div>
          </div>
        </div>
      </footer>

      <button
        type="button"
        onClick={() => setChatbotOpen(true)}
        className="fixed bottom-6 right-6 z-40"
        aria-label="AI 챗봇 열기"
      >
        <div className="shadow-lg rounded-full">
          <Button className="rounded-full h-12 w-12 p-0">
            <MessageSquare className="h-5 w-5" />
          </Button>
        </div>
      </button>

      <ChatbotModal
        open={chatbotOpen}
        onClose={() => setChatbotOpen(false)}
        isAuthenticated={isAuthenticated}
        onGoLogin={() => navigateTo("login", undefined, true)}
      />

      <RecommendedBidsModal
        open={recommendationsOpen}
        onClose={closeRecommendationsPermanently}
        bids={recommendedBids}
        onView={handleViewRecommended}
        onAddToCart={handleAddToCart}
      />
    </div>
  );
}