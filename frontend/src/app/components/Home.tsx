import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api/auth";
import { fetchWishlist } from "../api/wishlist";
import { mask_name } from "../utils/masking";

type AuthUser = {
	name: string;
	email?: string;
};

export function Home() {
	const navigate = useNavigate();

	const [wishlistCount, setWishlistCount] = useState(0);

	// 홈 우측 로그인 폼
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");

	const [errorMsg, setErrorMsg] = useState<string | null>(null);
	const [submitting, setSubmitting] = useState(false);

	const [isAuthed, setIsAuthed] = useState(() => {
		const uid = localStorage.getItem("userId");
		return !!uid && uid !== "undefined";
	});

	const [user, setUser] = useState<AuthUser | null>(() => {
		const uid = localStorage.getItem("userId");
		if (!uid || uid === "undefined") return null;

		const name =
			localStorage.getItem("userName") || localStorage.getItem("name");
		const storedEmail = localStorage.getItem("email") || undefined;

		return { name: name || "사용자", email: storedEmail };
	});

	useEffect(() => {
		if (!isAuthed) {
			setWishlistCount(0);
			return;
		}

		const uidStr = localStorage.getItem("userId");
		const uid = Number(uidStr);
		if (!Number.isFinite(uid)) return;

		fetchWishlist(uid)
			.then((items) => setWishlistCount(items.length))
			.catch(() => setWishlistCount(0));
	}, [isAuthed]);

	const onQuickLogin = async () => {
		setErrorMsg(null);
		if (!email.trim() || !password.trim()) {
			setErrorMsg("이메일과 비밀번호를 입력하세요.");
			return;
		}
		try {
			setSubmitting(true);
			const res = await login(email.trim(), password);

			if (res.status !== "success" || !res.data) {
				setErrorMsg(res.message || "로그인 실패");
				return;
			}

			const id = (res as any)?.data?.id;

			if (typeof id === "number" && Number.isFinite(id)) {
				localStorage.setItem("userId", String(id));
			} else {
				localStorage.removeItem("userId");
				setErrorMsg(
					"로그인 정보 처리 중 문제가 발생했습니다. 다시 시도해주세요."
				);
				return;
			}

			localStorage.setItem("userName", String(res.data?.name ?? ""));
			localStorage.setItem("email", String(res.data?.email ?? email.trim()));

			const anyData = res.data as any;
			if (anyData?.refreshToken)
				localStorage.setItem("refreshToken", String(anyData.refreshToken));

			setIsAuthed(true);
			setUser({
				name: String(res.data?.name ?? "사용자"),
				email: String(res.data?.email ?? email.trim()),
			});
			navigate("/", { replace: true });
		} catch (e: any) {
			setErrorMsg(e?.message || "로그인 중 오류가 발생했습니다.");
		} finally {
			setSubmitting(false);
		}
	};

	const onLogout = () => {
		localStorage.removeItem("userId");
		localStorage.removeItem("refreshToken");
		localStorage.removeItem("userName");
		localStorage.removeItem("name");
		localStorage.removeItem("email");
		window.location.href = "/";
	};

	return (
		<div className="bg-slate-50">
			<div className="max-w-7xl mx-auto px-5 py-8">
				<div className="flex items-end justify-between gap-4 mb-6">
					<div>
						<div className="text-sm text-slate-500">AI 기반 입찰 탐색 · 관리</div>
						<h1 className="text-2xl font-bold tracking-tight text-slate-900">
							공고를 더 빠르게 찾고, 더 확실하게 준비하세요
						</h1>
					</div>
				</div>

				{/* ✅ 1행: 바로가기(좌) + 로그인/프로필(우) => 같은 row라서 높이 매칭 가능 */}
				<div className="grid grid-cols-12 gap-6 items-stretch">
					{/* LEFT - 바로가기 */}
					<div className="col-span-12 lg:col-span-8 h-full">
						<section className="bg-white border rounded-2xl p-5 shadow-sm h-full">
							<div className="flex items-center justify-between mb-4">
								<h2 className="font-semibold text-slate-900">바로가기</h2>
								<div className="text-sm text-slate-500">
									핵심 기능 4개를 빠르게 접근
								</div>
							</div>

							<div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
								<QuickBox
									title="대시보드"
									desc="지표/분포/현황 한눈에"
									onClick={() => navigate("/dashboard")}
								/>
								<QuickBox
									title="공고 찾기"
									desc="조건 필터 & AI 검색"
									onClick={() => navigate("/bids")}
								/>
								<QuickBox
									title="장바구니"
									desc={`관심 공고 관리${
										wishlistCount ? ` · ${wishlistCount}건` : ""
									}`}
									badge={wishlistCount}
									onClick={() => navigate("/cart")}
								/>
								<QuickBox
									title="커뮤니티"
									desc="실무 팁/질문/공유"
									onClick={() => navigate("/community")}
								/>
							</div>
						</section>
					</div>

					{/* RIGHT - 로그인/프로필 */}
					<div className="col-span-12 lg:col-span-4 h-full">
						{!isAuthed ? (
							<aside className="bg-white border rounded-2xl p-5 shadow-sm h-full flex flex-col">
								<h3 className="text-lg font-semibold text-slate-900 mb-2">
									로그인
								</h3>
								<p className="text-sm text-slate-500 mb-4">
									로그인하면 장바구니/알림/AI 기능을 이용할 수 있습니다.
								</p>

								<div className="space-y-3">
									<div>
										<div className="text-sm font-medium text-slate-700 mb-1">
											이메일
										</div>
										<input
											value={email}
											onChange={(e) => setEmail(e.target.value)}
											className="w-full h-11 rounded-xl bg-slate-50 border px-3 outline-none focus:ring-4 focus:ring-blue-100 focus:border-blue-300"
											placeholder="name@company.com"
											type="email"
										/>
									</div>

									<div>
										<div className="text-sm font-medium text-slate-700 mb-1">
											비밀번호
										</div>
										<input
											value={password}
											onChange={(e) => setPassword(e.target.value)}
											className="w-full h-11 rounded-xl bg-slate-50 border px-3 outline-none focus:ring-4 focus:ring-blue-100 focus:border-blue-300"
											placeholder="••••••••"
											type="password"
										/>
									</div>

									{errorMsg && (
										<div className="text-sm text-red-600">{errorMsg}</div>
									)}

									<button
										disabled={submitting}
										onClick={onQuickLogin}
										className="w-full h-11 rounded-xl bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-60"
									>
										{submitting ? "로그인 중..." : "로그인"}
									</button>

									<div className="grid grid-cols-1 gap-3">
										<button
											onClick={() => navigate("/signup")}
											className="h-11 rounded-xl border hover:bg-slate-50"
										>
											회원가입
										</button>
									</div>
								</div>

								{/* 아래 링크는 카드 하단 고정 느낌 */}
								<div className="mt-auto pt-4 flex justify-between text-sm text-slate-500">
									<button
										onClick={() => navigate("/find-account")}
										className="hover:text-blue-600 hover:underline"
									>
										계정 찾기
									</button>
									<button
										onClick={() => navigate("/reset-password")}
										className="hover:text-blue-600 hover:underline"
									>
										비밀번호 찾기
									</button>
								</div>
							</aside>
						) : (
							<aside className="bg-white border rounded-2xl p-5 shadow-sm h-full flex flex-col">
								<div className="flex items-center justify-between mb-4">
									<div>
										<div className="text-sm text-slate-500">환영합니다</div>
										<div className="text-lg font-semibold text-slate-900">
											{mask_name(user?.name ?? "사용자")}
										</div>
										{user?.email && (
											<div className="text-sm text-slate-500">{user.email}</div>
										)}
									</div>
									<div className="w-11 h-11 rounded-xl bg-blue-600 text-white flex items-center justify-center font-bold">
										{user?.name?.slice(0, 1) ?? "U"}
									</div>
								</div>

								<div className="space-y-2">
									<button
										onClick={() => navigate("/profile")}
										className="w-full h-11 rounded-xl bg-slate-900 text-white hover:bg-slate-800"
									>
										프로필 수정
									</button>
									<button
										onClick={() => navigate("/notifications")}
										className="w-full h-11 rounded-xl border hover:bg-slate-50"
									>
										알림
									</button>
									<button
										onClick={onLogout}
										className="w-full h-11 rounded-xl border hover:bg-slate-50 text-slate-700"
									>
										로그아웃
									</button>
								</div>
							</aside>
						)}
					</div>

					{/* ✅ 2행: 오늘의 추천(좌만), 우측은 빈칸으로 둬서 레이아웃 깔끔 */}
					<div className="col-span-12 lg:col-span-8">
						<section className="bg-white border rounded-2xl p-5 shadow-sm">
							<div className="flex items-center justify-between mb-3">
								<h3 className="font-semibold text-slate-900">오늘의 추천</h3>
								<button
									className="text-sm text-blue-600 hover:underline"
									onClick={() => navigate("/bids")}
								>
									더 보기
								</button>
							</div>

							<div className="grid grid-cols-1 md:grid-cols-3 gap-4">
								<MiniStat label="신규 공고" value="67" sub="이번 달" />
								<MiniStat label="마감 임박" value="8" sub="3일 이내" />
								<MiniStat
									label="관심 공고"
									value={String(wishlistCount)}
									sub="장바구니"
								/>
							</div>

							<div className="mt-4 text-sm text-slate-500">
								상단 AI 검색창에서 자연어로 조건을 입력하면 공고 탐색 흐름으로 바로
								연결됩니다.
							</div>
						</section>
					</div>
				</div>
			</div>
		</div>
	);
}

function QuickBox({
	title,
	desc,
	onClick,
	badge,
}: {
	title: string;
	desc: string;
	onClick: () => void;
	badge?: number;
}) {
	return (
		<button
			onClick={onClick}
			className="group relative text-left border rounded-2xl p-4 hover:border-blue-200 hover:bg-blue-50/40 transition"
		>
			<div className="flex items-center justify-between mb-1">
				<div className="font-semibold text-slate-900">{title}</div>
				{!!badge && badge > 0 && (
					<span className="min-w-[22px] h-[22px] px-1 rounded-full bg-slate-900 text-white text-[12px] flex items-center justify-center">
						{badge}
					</span>
				)}
			</div>
			<div className="text-sm text-slate-500">{desc}</div>

			<div className="absolute right-4 bottom-4 text-sm text-blue-600 opacity-0 group-hover:opacity-100 transition">
				이동 →
			</div>
		</button>
	);
}

function MiniStat({
	label,
	value,
	sub,
}: {
	label: string;
	value: string;
	sub: string;
}) {
	return (
		<div className="border rounded-2xl p-4 bg-slate-50">
			<div className="text-sm text-slate-500">{label}</div>
			<div className="text-2xl font-bold text-slate-900">{value}</div>
			<div className="text-sm text-slate-500">{sub}</div>
		</div>
	);
}
