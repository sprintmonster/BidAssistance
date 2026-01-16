import { useEffect, useState } from "react";
import type { Page } from "../types/navigation";

/* ===== Pages ===== */
import { Login } from "./components/Login";
import { Register } from "./components/Register";
import { BidDiscovery } from "./components/BidDiscovery";
import { CartPage } from "./components/CartPage";
import { Dashboard } from "./components/Dashboard";
import { CommunityPage } from "./components/CommunityPage";
import { CommunityBoard } from "./components/CommunityBoard";
import { ChatbotPage } from "./components/ChatbotPage";
import { BidSummary } from "./components/BidSummary";

/* ===== UI ===== */
import { Toast } from "./components//ui/Toast";
import { useToast } from "./components//ui/useToast";

export default function App() {
  /* ===== 인증 ===== */
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(
    !!localStorage.getItem("accessToken")
  );

  /* ===== 라우팅 ===== */
  const [currentPage, setCurrentPage] = useState<Page>("dashboard");
  const [selectedBidId, setSelectedBidId] = useState<number | null>(null);

  /* ===== 전역 상태 ===== */
  const [initializing, setInitializing] = useState(true);
  const [globalLoading, setGlobalLoading] = useState(false);

  /* ===== 토스트 ===== */
  const { toast, showToast } = useToast();

  /* =========================
     초기화
  ========================= */
  useEffect(() => {
    setInitializing(false);
  }, []);

  /* =========================
     토큰 만료 감지 (401)
  ========================= */
  useEffect(() => {
    const handleStorage = () => {
      if (!localStorage.getItem("accessToken")) {
        setIsAuthenticated(false);
        setCurrentPage("dashboard");
        showToast("로그인이 만료되었습니다", "error");
      }
    };

    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, [showToast]);

  /* =========================
     네비게이션
  ========================= */
  const navigate = (page: Page, bidId?: number) => {
    if (bidId) setSelectedBidId(bidId);
    setCurrentPage(page);
  };

  const logout = () => {
    localStorage.removeItem("accessToken");
    setIsAuthenticated(false);
    setCurrentPage("dashboard");
    showToast("로그아웃 되었습니다", "success");
  };

  /* =========================
     초기 로딩
  ========================= */
  if (initializing) {
    return (
      <div className="h-screen flex items-center justify-center">
        초기화 중...
      </div>
    );
  }

  /* =========================
     비로그인 상태
  ========================= */
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Login onSuccess={() => setIsAuthenticated(true)} />
      </div>
    );
  }

  /* =========================
     로그인 상태
  ========================= */
  return (
    <div className="min-h-screen flex flex-col">
      {/* ===== Header ===== */}
      <header className="border-b px-6 py-3 flex justify-between items-center">
        <div className="font-bold text-lg">입찰 플랫폼</div>

        <nav className="flex gap-4 text-sm">
          <button onClick={() => navigate("dashboard")}>대시보드</button>
          <button onClick={() => navigate("bids")}>공고</button>
          <button onClick={() => navigate("cart")}>장바구니</button>
          <button onClick={() => navigate("community")}>커뮤니티</button>
          <button onClick={() => navigate("chatbot")}>AI 분석</button>
          <button onClick={logout}>로그아웃</button>
        </nav>
      </header>

      {/* ===== Main ===== */}
      <main className="flex-1 p-6">
        {currentPage === "dashboard" && <Dashboard />}

        {currentPage === "bids" && (
          <BidDiscovery
            onNavigate={navigate}
            setGlobalLoading={setGlobalLoading}
            showToast={showToast}
          />
        )}

        {currentPage === "summary" && selectedBidId !== null && (
          <BidSummary bidId={selectedBidId} />
        )}

        {currentPage === "cart" && (
          <CartPage
            onNavigate={navigate}
            setGlobalLoading={setGlobalLoading}
            showToast={showToast}
          />
        )}

        {currentPage === "community" && (
          <CommunityPage onNavigate={navigate} />
        )}

        {currentPage === "community-board" && (
          <CommunityBoard onNavigate={navigate} />
        )}

        {currentPage === "chatbot" && (
          <ChatbotPage />
        )}
      </main>

      {/* ===== Global Loading ===== */}
      {globalLoading && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
          <div className="bg-white px-6 py-3 rounded">
            처리 중...
          </div>
        </div>
      )}

      {/* ===== Toast ===== */}
      {toast && <Toast message={toast.message} type={toast.type} />}
    </div>
  );
}
