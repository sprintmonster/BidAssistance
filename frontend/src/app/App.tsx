import { BrowserRouter, Routes, Route, useNavigate } from "react-router-dom";
import { useState } from "react";

import { AppLayout } from "./layout/AppLayout";
import { PageContainer } from "./layout/PageContainer";

/* Pages/Components */
import { HomePage } from "./components/HomePage";
import { LoginPage } from "./components/LoginPage";
import { SignupPage } from "./components/SignupPage";
import { FindAccountPage } from "./components/FindAccount";
import { ResetPasswordPage } from "./components/ResetPasswordPage";

import { Dashboard } from "./components/Dashboard";
import { BidDiscovery } from "./components/BidDiscovery";
import { CartPage } from "./components/CartPage";
import { CommunityPage } from "./components/CommunityPage";

/* Toast */
import { Toast } from "./components/ui/Toast";
import { useToast } from "./components/ui/useToast";

function SignupRoute() {
	const navigate = useNavigate();

	// NOTE: SignupPage는 현재 props 기반이라 wrapper로 연결.
	// 실제 회원가입 API 연동은 SignupPage 내부/또는 여기에서 register()를 붙이면 됨.
	return (
		<SignupPage
			onSignup={() => navigate("/login")}
			onNavigateToLogin={() => navigate("/login")}
		/>
	);
}

function FindAccountRoute() {
	const navigate = useNavigate();

	return (
		<FindAccountPage
			onFindAccount={async () => {
				// TODO: 계정 찾기 API 연동 시 여기서 호출
			}}
			onNavigateToLogin={() => navigate("/login")}
		/>
	);
}

function ResetPasswordRoute() {
	const navigate = useNavigate();

	return <ResetPasswordPage onNavigateToLogin={() => navigate("/login")} />;
}

export default function App() {
	const [globalLoading, setGlobalLoading] = useState(false);
	const { toast, showToast } = useToast();

	return (
		<BrowserRouter>
			{/* 전역 로딩/토스트는 라우트와 무관하게 유지 */}
			{globalLoading && (
				<div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
					<div className="bg-white px-6 py-3 rounded">처리 중...</div>
				</div>
			)}
			{toast && <Toast message={toast.message} type={toast.type} />}

			<Routes>
				{/* ✅ AppLayout 적용 구간 (홈/대시보드/기능페이지) */}
				<Route element={<AppLayout />}>
					<Route path="/" element={<HomePage />} />

					<Route
						path="/dashboard"
						element={
							<PageContainer>
								<Dashboard />
							</PageContainer>
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
							<PageContainer>
								<CartPage
									setGlobalLoading={setGlobalLoading}
									showToast={showToast}
								/>
							</PageContainer>
						}
					/>
					<Route
						path="/community"
						element={
							<PageContainer>
								<CommunityPage />
							</PageContainer>
						}
					/>
				</Route>

				{/* ✅ AppLayout 미적용 구간 (인증 페이지는 “원래 UI”처럼 단독 화면) */}
				<Route path="/login" element={<LoginPage />} />
				<Route path="/signup" element={<SignupRoute />} />
				<Route path="/find-account" element={<FindAccountRoute />} />
				<Route path="/reset-password" element={<ResetPasswordRoute />} />
			</Routes>
		</BrowserRouter>
	);
}
