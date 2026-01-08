import { useCallback, useEffect, useState } from "react";
import { LoginPage } from "./components/LoginPage";
import { SignupPage } from "./components/SignupPage";
import { Megaphone } from "lucide-react";

import { Dashboard } from "./components/Dashboard";
import { BidDiscovery } from "./components/BidDiscovery";
import { BidSummary } from "./components/BidSummary";
import { CartPage } from "./components/CartPage";
import { NotificationsPage, type NotificationItem } from "./components/NotificationsPage";
import { ChatbotPage } from "./components/ChatbotPage";
import { ProfilePage } from "./components/ProfilePage";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import type { Page } from "../types/navigation";

import { FindAccountPage } from "./components/FindAccount";
import { ResetPasswordPage } from "./components/ResetPasswordPage";
import { NoticePage } from "./components/NoticePage";

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
} from "lucide-react";
import { toast, Toaster } from "sonner";

type NavState = { page: Page; bidId?: number };

function isNavState(v: unknown): v is NavState {
    if (!v || typeof v !== "object") return false;
    const anyV = v as Record<string, unknown>;
    return typeof anyV.page === "string";
}

const DEFAULT_NOTIFICATIONS: NotificationItem[] = [
    {
        id: 1,
        type: "deadline",
        title: "마감 임박 알림",
        message: "서울시 강남구 도로 보수공사가 2일 후 마감됩니다",
        time: "2시간 전",
        read: false,
        urgent: true,
    },
    {
        id: 2,
        type: "correction",
        title: "정정공고 발표",
        message: "경기도 성남시 공공건물 신축공사의 예산이 87억원에서 92억원으로 변경되었습니다",
        time: "5시간 전",
        read: false,
        urgent: false,
    },
    {
        id: 3,
        type: "reannouncement",
        title: "재공고 등록",
        message: "인천 항만시설 보수공사가 재공고 되었습니다",
        time: "1일 전",
        read: true,
        urgent: false,
    },
    {
        id: 4,
        type: "unsuccessful",
        title: "유찰 공고",
        message: "부산시 해운대구 주차장 건설이 유찰되었습니다. 재입찰 예정",
        time: "1일 전",
        read: true,
        urgent: false,
    },
    {
        id: 5,
        type: "new",
        title: "신규 공고",
        message: "관심 지역(서울)에 새로운 공고 3건이 등록되었습니다",
        time: "2일 전",
        read: true,
        urgent: false,
    },
    {
        id: 6,
        type: "deadline",
        title: "마감 임박 알림",
        message: "인천광역시 연수구 학교시설 개선공사가 4일 후 마감됩니다",
        time: "2일 전",
        read: true,
        urgent: false,
    },
    {
        id: 7,
        type: "correction",
        title: "정정공고 발표",
        message: "대전시 유성구 복지센터 리모델링의 기술요건이 변경되었습니다",
        time: "3일 전",
        read: true,
        urgent: false,
    },
];

function loadNotifications(): NotificationItem[] {
    const raw = localStorage.getItem("notifications");
    if (!raw) return DEFAULT_NOTIFICATIONS;
    try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) return parsed as NotificationItem[];
        return DEFAULT_NOTIFICATIONS;
    } catch {
        return DEFAULT_NOTIFICATIONS;
    }
}

export default function App() {
    const [currentPage, setCurrentPage] = useState<Page>("login");
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [userEmail, setUserEmail] = useState("");
    const [cartItems, setCartItems] = useState<number[]>([]);
    const [selectedBidId, setSelectedBidId] = useState<number | undefined>();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    const [notifications, setNotifications] = useState<NotificationItem[]>(() => loadNotifications());

    useEffect(() => {
        localStorage.setItem("notifications", JSON.stringify(notifications));
    }, [notifications]);

    const unreadCount = notifications.filter((n) => !n.read).length;

    const markAllNotificationsRead = useCallback(() => {
        setNotifications((prev) => {
            const hasUnread = prev.some((n) => !n.read);
            if (!hasUnread) return prev;
            return prev.map((n) => (n.read ? n : { ...n, read: true }));
        });
    }, []);

    const markNotificationRead = useCallback((id: number) => {
        setNotifications((prev) => {
            let changed = false;
            const next = prev.map((n) => {
                if (n.id !== id) return n;
                if (n.read) return n;
                changed = true;
                return { ...n, read: true };
            });
            return changed ? next : prev;
        });
    }, []);

    // ---- history sync ----
    useEffect(() => {
        const st = window.history.state;
        if (isNavState(st) && st.page) {
            setCurrentPage(st.page as Page);
            setSelectedBidId(st.bidId as number | undefined);
        } else {
            window.history.replaceState({ page: "login" } satisfies NavState, "");
        }

        const onPopState = (e: PopStateEvent) => {
            if (isNavState(e.state) && e.state.page) {
                setCurrentPage(e.state.page);
                setSelectedBidId(e.state.bidId);
                setMobileMenuOpen(false);
            }
        };

        window.addEventListener("popstate", onPopState);
        return () => window.removeEventListener("popstate", onPopState);
    }, []);

    const navigateTo = (page: Page, bidId?: number, replace: boolean = false) => {
        const next: NavState = { page, bidId };

        if (replace) window.history.replaceState(next, "");
        else window.history.pushState(next, "");

        setCurrentPage(page);
        setSelectedBidId(bidId);
        setMobileMenuOpen(false);
    };

    const handleLogin = (email: string) => {
        setIsAuthenticated(true);
        setUserEmail(email);
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

        navigateTo("login", undefined, true);
        toast.info("로그아웃되었습니다");
    };

    const handleNavigate = (page: Page, bidId?: number) => {
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

    const handleFindAccount = async (payload: {
        name: string;
        birthDate: string;
        questionKey: string;
        answer: string;
    }) => {
        console.log("find account payload:", payload);

        // 실제 서비스에선 여기서 API 호출하고,
        // 결과가 맞으면 마스킹 이메일 안내 or 이메일 발송 안내 등을 해요.
        toast.success("계정 찾기 요청이 접수되었습니다");
        navigateTo("login"); // 요청 후 로그인으로 복귀 (원하면 유지해도 됨)
    };

    const handleFindPassword = async (payload: {
        name: string;
        email: string;
        birthDate: string;
        questionKey: string;
        answer: string;
    }) => {
        console.log("find password payload:", payload);

        // 실제 서비스에선 여기서 API 호출하고,
        // 결과가 맞으면 마스킹 이메일 안내 or 이메일 발송 안내 등을 해요.
        toast.success("비밀번호 찾기 요청이 접수되었습니다");
        navigateTo("login"); // 요청 후 로그인으로 복귀 (원하면 유지해도 됨)
    };

    // ---- Auth pages ----
    if (!isAuthenticated) {
        if (currentPage === "signup") {
            return (
                <SignupPage
                    onSignup={handleSignup}
                    onNavigateToLogin={() => navigateTo("login")}
                />
            );

        }

        if (currentPage === "findAccount") {
            return (
                <FindAccountPage
                    onFindAccount={handleFindAccount}
                    onNavigateToLogin={() => navigateTo("login")}
                />
            );
        }
        if (currentPage === "resetPassword") {
            return (
                <ResetPasswordPage
                    onRequestReset={async (payload) => {
                        console.log("reset password payload:", payload);
                        toast.success("비밀번호 재설정 요청이 접수되었습니다");
                        navigateTo("login");
                    }}
                    onNavigateToLogin={() => navigateTo("login")}
                />
            );
        }


        return (
            <LoginPage
                onLogin={handleLogin}
                onNavigateToSignup={() => navigateTo("signup")}
                onNavigateToFindAccount={() => navigateTo("findAccount")}
                onNavigateToResetPassword={() => navigateTo("resetPassword")}
            />
        );
    }

    const menuItems = [
        { id: "dashboard" as Page, icon: LayoutDashboard, label: "대시보드" },
        { id: "bids" as Page, icon: Search, label: "공고 찾기" },
        { id: "notices" as Page, icon: Megaphone, label: "공지사항" }, // ✅ 추가
        { id: "cart" as Page, icon: ShoppingCart, label: "장바구니", badge: cartItems.length },
        { id: "notifications" as Page, icon: Bell, label: "알림", badge: unreadCount },
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
                {currentPage === "dashboard" && <Dashboard onNavigate={handleNavigate} cart={cartItems} />}
                {currentPage === "bids" && <BidDiscovery onNavigate={handleNavigate} onAddToCart={handleAddToCart} />}
                {currentPage === "notices" && <NoticePage onNavigate={function(page: Page, bidId?: number): void {
                    throw new Error("Function not implemented.");
                } } />}
                {currentPage === "summary" && <BidSummary bidId={selectedBidId} onNavigate={handleNavigate} />}
                {currentPage === "cart" && (
                    <CartPage cartItems={cartItems} onRemoveFromCart={handleRemoveFromCart} onNavigate={handleNavigate} />
                )}
                {currentPage === "notifications" && (
                    <NotificationsPage
                        onNavigate={handleNavigate}
                        notifications={notifications}
                        onMarkRead={markNotificationRead}
                        onMarkAllRead={markAllNotificationsRead}
                        autoMarkAllReadOnEnter={true}
                    />
                )}
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
