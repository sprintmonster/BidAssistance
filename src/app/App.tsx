import { useState } from "react";
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
  X
} from "lucide-react";
import { toast, Toaster } from "sonner";

type Page = "login" | "signup" | "dashboard" | "bids" | "analytics" | "summary" | "cart" | "notifications" | "chatbot" | "profile";

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("login");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userEmail, setUserEmail] = useState("");
  const [cartItems, setCartItems] = useState<number[]>([]);
  const [selectedBidId, setSelectedBidId] = useState<number | undefined>();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleLogin = (email: string) => {
    setIsAuthenticated(true);
    setUserEmail(email);
    setCurrentPage("dashboard");
    toast.success("로그인되었습니다");
  };

  const handleSignup = (email: string) => {
    setIsAuthenticated(true);
    setUserEmail(email);
    setCurrentPage("dashboard");
    toast.success("회원가입이 완료되었습니다");
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserEmail("");
    setCurrentPage("login");
    setCartItems([]);
    toast.info("로그아웃되었습니다");
  };

  const handleNavigate = (page: Page, bidId?: number) => {
    setCurrentPage(page);
    if (bidId !== undefined) {
      setSelectedBidId(bidId);
    }
    setMobileMenuOpen(false);
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
    setCartItems(cartItems.filter(id => id !== bidId));
    toast.success("장바구니에서 제거되었습니다");
  };

  if (!isAuthenticated) {
    if (currentPage === "signup") {
      return <SignupPage onSignup={handleSignup} onNavigateToLogin={() => setCurrentPage("login")} />;
    }
    return <LoginPage onLogin={handleLogin} onNavigateToSignup={() => setCurrentPage("signup")} />;
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
                <p className="text-xs text-muted-foreground hidden sm:block">Smart Procurement Platform</p>
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
        {currentPage === "dashboard" && <Dashboard onNavigate={handleNavigate} />}
        {currentPage === "bids" && (
          <BidDiscovery onNavigate={handleNavigate} onAddToCart={handleAddToCart} />
        )}
        {currentPage === "analytics" && <AnalyticsReport />}
        {currentPage === "summary" && <BidSummary bidId={selectedBidId} onNavigate={handleNavigate} />}
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
            <p className="text-sm text-muted-foreground">
              © 2026 입찰 인텔리전스. All rights reserved.
            </p>
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
