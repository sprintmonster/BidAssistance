import { useEffect, useMemo, useState } from "react";
import { LoginPage } from "./components/LoginPage";
import { SignupPage } from "./components/SignupPage";
import { Dashboard } from "./components/Dashboard";
import { BidDiscovery } from "./components/BidDiscovery";
import { AnalyticsReport } from "./components/AnalyticsReport";
import { BidSummary } from "./components/BidSummary";
import { CartPage } from "./components/CartPage";
import { NotificationsPage } from "./components/NotificationsPage";
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
  TrendingUp,
  ShoppingCart,
  Bell,
  MessageSquare,
  User,
  Building2,
  LogOut,
  Menu,
  X,
  Sparkles,
  LogIn,
} from "lucide-react";
import { toast, Toaster } from "sonner";

/**
 * 중요:
 * Page 타입에 "home"이 포함되어 있어야 합니다.
 * (../types/navigation.ts에서 Page 유니온에 "home" 추가)
 */
type NavState = { page: Page; bidId?: number };

function isNavState(v: unknown): v is NavState {
  if (!v || typeof v !== "object") return false;
  const anyV = v as Record<string, unknown>;
  return typeof anyV.page === "string";
}

function isPublicPage(page: Page) {
  return page === "home" || page === "login" || page === "signup";
}

function HomeLanding({
  onGoLogin,
  onGoSignup,
  onNavigate,
}: {
  onGoLogin: () => void;
  onGoSignup: () => void;
  onNavigate: (page: Page) => void;
}) {
  const [q, setQ] = useState("");

  const quickLinks = useMemo(
    () => [
      { id: "dashboard" as Page, label: "대시보드", icon: LayoutDashboard },
      { id: "bids" as Page, label: "공고 찾기", icon: Search },
      { id: "analytics" as Page, label: "낙찰 분석", icon: TrendingUp },
      { id: "cart" as Page, label: "장바구니", icon: ShoppingCart },
      { id: "notifications" as Page, label: "알림", icon: Bell },
      { id: "chatbot" as Page, label: "AI 챗봇", icon: MessageSquare },
      { id: "profile" as Page, label: "마이페이지", icon: User },
    ],
    []
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <Toaster position="top-right" />

      <header className="bg-white border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <Building2 className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold">입찰 인텔리전스</h1>
                <p className="text-xs text-muted-foreground hidden sm:block">
                  Smart Procurement Platform
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button variant="outline" onClick={onGoLogin} className="gap-2">
                <LogIn className="h-4 w-4" />
                로그인
              </Button>
              <Button onClick={onGoSignup}>회원가입</Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-12 gap-6">
          {/* 중앙 */}
          <div className="col-span-12 lg:col-span-8 space-y-6">
            {/* 메뉴로 이동하는 박스들 */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {quickLinks.map((x) => {
                const Icon = x.icon;
                return (
                  <button key={x.id} onClick={() => onNavigate(x.id)} className="text-left">
                    <Card className="hover:shadow-sm transition">
                      <CardContent className="p-4 flex items-center gap-3">
                        <div className="h-9 w-9 rounded-md bg-gray-100 flex items-center justify-center">
                          <Icon className="h-4 w-4" />
                        </div>
                        <div className="font-medium">{x.label}</div>
                      </CardContent>
                    </Card>
                  </button>
                );
              })}
            </div>

            {/* AI 검색 패널 */}
            <Card>
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
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  placeholder='예: "서울/경기 10억~50억 시설공사, 마감 임박 우선"'
                />
                <div className="flex gap-2">
                  <Button
                    className="gap-2"
                    onClick={() => {
                      if (!q.trim()) {
                        toast.info("검색어를 입력해 주세요.");
                        return;
                      }
                      // 비로그인 상태에서 바로 AI 기능 접근은 막히므로,
                      // 사용자는 요구사항대로 로그인 페이지로 이동하게 됩니다.
                      // (실제 이동/차단 로직은 App.handleNavigate에서 수행)
                      localStorage.setItem("chatbot.initialQuery", q.trim());
                      onNavigate("chatbot");
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

          {/* 우측: 로그인 박스 */}
          <div className="col-span-12 lg:col-span-4 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <LogIn className="h-5 w-5" />
                  로그인
                </CardTitle>
                <CardDescription>
                  로그인하면 공고/장바구니/알림/AI 기능을 이용할 수 있습니다.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full" onClick={onGoLogin}>
                  로그인 페이지로 이동
                </Button>
                <Button className="w-full" variant="outline" onClick={onGoSignup}>
                  회원가입
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-sm text-muted-foreground">© 2026 입찰 인텔리전스. All rights reserved.</p>
            <div className="flex gap-4 text-sm text-muted-foreground">
              <a href="#" className="hover:text-blue-600">이용약관</a>
              <a href="#" className="hover:text-blue-600">개인정보처리방침</a>
              <a href="#" className="hover:text-blue-600">고객지원</a>
            </div>
          </div>
        </div>
      </footer>
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

  const navigateTo = (page: Page, bidId?: number, replace: boolean = false) => {
    const next: NavState = { page, bidId };

    if (replace) window.history.replaceState(next, "");
    else window.history.pushState(next, "");

    setCurrentPage(page);
    setSelectedBidId(bidId);
    setMobileMenuOpen(false);
  };

  // ---- history sync (핵심) ----
  useEffect(() => {
    const st = window.history.state;

    // 최초 로드: state가 없으면 home으로 초기화
    if (isNavState(st) && st.page) {
      // 보호 페이지를 비로그인 상태에서 직접 진입/새로고침한 경우 → 로그인으로 강제
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

      // 브라우저 뒤/앞으로 보호 페이지 접근도 차단
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

  const handleLogin = (email: string) => {
    setIsAuthenticated(true);
    setUserEmail(email);
    // 로그인 후에는 login 히스토리를 dashboard로 "대체"해서 back이 login으로 가지 않게 함
    navigateTo("dashboard", undefined, true);
    toast.success("로그인되었습니다");
  };

  const handleSignup = (email: string) => {
    setIsAuthenticated(true);
    setUserEmail(email);
    navigateTo("dashboard", undefined, true);
    toast.success("회원가입이 완료되었습니다");
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserEmail("");
    setCartItems([]);
    setSelectedBidId(undefined);

    // 로그아웃 시 현재 엔트리를 login으로 대체
    navigateTo("login", undefined, true);
    toast.info("로그아웃되었습니다");
  };

  const handleNavigate = (page: Page, bidId?: number) => {
    // 비로그인 상태: home/login/signup만 허용
    if (!isAuthenticated && !isPublicPage(page)) {
      toast.info("로그인이 필요합니다.");
      // replace 권장: 막힌 이동을 히스토리에 남기지 않음
      navigateTo("login", undefined, true);
      return;
    }
    navigateTo(page, bidId, false);
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

  // ---- 비로그인: 기존 로그인/회원가입 페이지는 그대로 사용 ----
  if (!isAuthenticated) {
    if (currentPage === "signup") {
      return (
        <SignupPage
          onSignup={handleSignup}
          onNavigateToLogin={() => navigateTo("login")}
        />
      );
    }
    if (currentPage === "login") {
      return (
        <LoginPage
          onLogin={handleLogin}
          onNavigateToSignup={() => navigateTo("signup")}
        />
      );
    }

    // home은 비로그인이어도 보여준다
    return (
      <HomeLanding
        onGoLogin={() => navigateTo("login")}
        onGoSignup={() => navigateTo("signup")}
        onNavigate={(p) => handleNavigate(p)}
      />
    );
  }

  const menuItems = [
    { id: "dashboard" as Page, icon: LayoutDashboard, label: "대시보드" },
    { id: "bids" as Page, icon: Search, label: "공고 찾기" },
    { id: "analytics" as Page, icon: TrendingUp, label: "낙찰 분석" },
    { id: "cart" as Page, icon: ShoppingCart, label: "장바구니", badge: cartItems.length },
    { id: "notifications" as Page, icon: Bell, label: "알림", badge: 2 },
    { id: "chatbot" as Page, icon: MessageSquare, label: "AI 챗봇" },
    { id: "profile" as Page, icon: User, label: "마이페이지" },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <Toaster position="top-right" />

      {/* Header */}
      <header className="bg-white border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <Building2 className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold">입찰 인텔리전스</h1>
                <p className="text-xs text-muted-foreground hidden sm:block">
                  Smart Procurement Platform
                </p>
              </div>
            </div>

            {/* Desktop Navigation */}
            <nav className="hidden lg:flex items-center gap-1">
              {menuItems.map((item) => {
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
                    {item.badge && item.badge > 0 && (
                      <Badge className="ml-2 h-5 w-5 flex items-center justify-center p-0 text-xs">
                        {item.badge}
                      </Badge>
                    )}
                  </Button>
                );
              })}
              <Button variant="ghost" size="sm" onClick={handleLogout}>
                <LogOut className="h-4 w-4 mr-2" />
                로그아웃
              </Button>
            </nav>

            {/* Mobile Menu Button */}
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

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="lg:hidden border-t bg-white">
            <nav className="px-4 py-2 space-y-1">
              {menuItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentPage === item.id;
                return (
                  <Button
                    key={item.id}
                    variant={isActive ? "default" : "ghost"}
                    className="w-full justify-start relative"
                    onClick={() => handleNavigate(item.id)}
                  >
                    <Icon className="h-4 w-4 mr-3" />
                    {item.label}
                    {item.badge && item.badge > 0 && (
                      <Badge className="ml-auto h-5 w-5 flex items-center justify-center p-0 text-xs">
                        {item.badge}
                      </Badge>
                    )}
                  </Button>
                );
              })}
              <Button variant="ghost" className="w-full justify-start" onClick={handleLogout}>
                <LogOut className="h-4 w-4 mr-3" />
                로그아웃
              </Button>
            </nav>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentPage === "home" && (
          <HomeLanding
            onGoLogin={() => navigateTo("login")}
            onGoSignup={() => navigateTo("signup")}
            onNavigate={(p) => handleNavigate(p)}
          />
        )}

        {currentPage === "dashboard" && (
          <Dashboard onNavigate={handleNavigate} cart={cartItems} />
        )}
        {currentPage === "bids" && (
          <BidDiscovery onNavigate={handleNavigate} onAddToCart={handleAddToCart} />
        )}
        {currentPage === "analytics" && <AnalyticsReport />}
        {currentPage === "summary" && (
          <BidSummary bidId={selectedBidId} onNavigate={handleNavigate} />
        )}
        {currentPage === "cart" && (
          <CartPage
            cartItems={cartItems}
            onRemoveFromCart={handleRemoveFromCart}
            onNavigate={handleNavigate}
          />
        )}
        {currentPage === "notifications" && <NotificationsPage onNavigate={handleNavigate} />}
        {currentPage === "chatbot" && <ChatbotPage />}
        {currentPage === "profile" && <ProfilePage userEmail={userEmail} />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-sm text-muted-foreground">© 2026 입찰 인텔리전스. All rights reserved.</p>
            <div className="flex gap-4 text-sm text-muted-foreground">
              <a href="#" className="hover:text-blue-600">이용약관</a>
              <a href="#" className="hover:text-blue-600">개인정보처리방침</a>
              <a href="#" className="hover:text-blue-600">고객지원</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
