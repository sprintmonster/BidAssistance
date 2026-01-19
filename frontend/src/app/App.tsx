import { BrowserRouter, Routes, Route, useNavigate } from "react-router-dom";
import { useState } from "react";

import { AppLayout } from "./layout/AppLayout";
import { PageContainer } from "./layout/PageContainer";

import { HomePage } from "./components/HomePage";
import { LoginPage } from "./components/LoginPage";
import { SignupPage } from "./components/SignupPage";
import { FindAccountPage } from "./components/FindAccount";
import { ResetPasswordPage } from "./components/ResetPasswordPage";

import { Dashboard } from "./components/Dashboard";
import { BidDiscovery } from "./components/BidDiscovery";
import { CartPage } from "./components/CartPage";
import { CommunityPage } from "./components/CommunityPage";
import { NoticePage } from "./components/NoticePage";
import { NotificationsPage } from "./components/NotificationsPage";
import { ProfilePage } from "./components/ProfilePage";

import { ProtectedRoute } from "./routes/ProtectedRoute";

import { Toast } from "./components/ui/Toast";
import { useToast } from "./components/ui/useToast";

function SignupRoute() {
  const navigate = useNavigate();
  return (
    <SignupPage
      onSignup={() => navigate("/login")}
      onNavigateToLogin={() => navigate("/login")}
      onNavigateToHome={() => navigate("/")}
    />
  );
}

function FindAccountRoute() {
  const navigate = useNavigate();
  return (
    <FindAccountPage
      onFindAccount={async () => {}}
      onNavigateToLogin={() => navigate("/login")}
      onNavigateToHome={() => navigate("/")}
    />
  );
}

function ResetPasswordRoute() {
  const navigate = useNavigate();
  return (
    <ResetPasswordPage 
      onNavigateToLogin={() => navigate("/login")} 
      onNavigateToHome={() => navigate("/")}
    />
  );
}

export default function App() {
  const [globalLoading, setGlobalLoading] = useState(false);
  const { toast, showToast } = useToast();

  return (
    <BrowserRouter>
      {globalLoading && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-[9999]">
          <div className="bg-white px-6 py-3 rounded-lg shadow">처리 중...</div>
        </div>
      )}
      {toast && <Toast message={toast.message} type={toast.type} />}

      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<HomePage />} />

          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <PageContainer>
                  <Dashboard />
                </PageContainer>
              </ProtectedRoute>
            }
          />

          <Route
            path="/bids"
            element={
              <PageContainer>
                <BidDiscovery
                  setGlobalLoading={setGlobalLoading}
                  showToast={showToast}
                />
              </PageContainer>
            }
          />

          <Route
            path="/cart"
            element={
              <ProtectedRoute>
                <PageContainer>
                  <CartPage
                    setGlobalLoading={setGlobalLoading}
                    showToast={showToast}
                  />
                </PageContainer>
              </ProtectedRoute>
            }
          />

          {/* ✅ 커뮤니티: 로그인 없이 접근 가능 */}
          <Route
            path="/community"
            element={
              <PageContainer>
                <CommunityPage />
              </PageContainer>
            }
          />

          <Route
            path="/notice"
            element={
              <ProtectedRoute>
                <PageContainer>
                  <NoticePage />
                </PageContainer>
              </ProtectedRoute>
            }
          />
          <Route
            path="/notifications"
            element={
              <ProtectedRoute>
                <PageContainer>
                  <NotificationsPage />
                </PageContainer>
              </ProtectedRoute>
            }
          />
          <Route
            path="/profile"
            element={
              <ProtectedRoute>
                <PageContainer>
                  <ProfilePage />
                </PageContainer>
              </ProtectedRoute>
            }
          />
        </Route>

        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupRoute />} />
        <Route path="/find-account" element={<FindAccountRoute />} />
        <Route path="/reset-password" element={<ResetPasswordRoute />} />
      </Routes>
    </BrowserRouter>
  );
}
