import { Outlet, NavLink, useLocation, useNavigate } from "react-router-dom";
import {
	useEffect,
	useMemo,
	useState,
	type FormEvent,
	type ReactNode,
} from "react";
import { ChatbotFloatingButton } from "../components/ChatbotFloatingButton";
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

	const isAuthed = useMemo(
		() => !!localStorage.getItem("accessToken"),
		[location.pathname],
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

	useEffect(() => {
		setMobileOpen(false);
	}, [location.pathname]);

	const onSubmitSearch = (e: FormEvent) => {
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
			{/* ===== Header (Top Bar) ===== */}
			<header className="border-b bg-slate-900 text-white">
				<div className="h-16 w-full flex items-center">
					{/* 좌측: 사이드바와 같은 폭(연결감) - md 이상에서 고정 */}
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

					{/* 우측: 상단 액션 영역 */}
					<div className="flex-1 h-16 flex items-center justify-end gap-2 px-5">
						<HeaderPill onClick={() => navigate("/notice")} label="공지사항" />
						<HeaderPill onClick={() => navigate("/notifications")} label="알림" />
						{/* 상단바에는 로그인/로그아웃/마이 절대 안 올림 */}
					</div>
				</div>
			</header>

			{/* ===== Mobile Sidebar Overlay (홈 제외) ===== */}
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
							onLogout={logout}
							onNavigate={(to) => navigate(to)}
						/>
					</aside>
				</div>
			)}

			{/* ===== Home Only: Search Row ===== */}
			{isHome && (
				<section className="bg-white text-slate-900 border-b border-slate-200">
					<div className="w-full px-5 py-6">
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
			{isHome ? (
				<main className="flex-1">
					<Outlet />
				</main>
			) : (
				/* 세로: 헤더~푸터 사이를 flex-1로 꽉 채움
				   가로: 좌측은 사이드바(260px)가 화면 왼쪽부터 시작(빈 여백 제거)
				*/
				<div className="flex-1 min-h-0 flex">
					<aside className="hidden md:flex w-[260px] shrink-0 bg-slate-900 text-white border-r border-white/10">
						<SideBarContent
							items={sideItems}
							pathname={location.pathname}
							isAuthed={isAuthed}
							onLogout={logout}
							onNavigate={(to) => navigate(to)}
						/>
					</aside>

					<main className="flex-1 min-w-0 min-h-0 bg-slate-50">
						<div className="h-full min-h-0 p-6">
							<Outlet />
						</div>
					</main>
				</div>
			)}

			{/* ===== Footer ===== */}
			<footer className="border-t bg-white py-5 text-sm text-gray-500">
				<div className="w-full px-5 flex items-center justify-between">
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
				<div className="text-xs font-semibold text-slate-300">메뉴</div>
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
				active
					? "bg-white/10 text-white"
					: "text-slate-200 hover:bg-white/5 hover:text-white",
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
