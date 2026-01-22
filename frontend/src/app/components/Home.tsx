import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { login } from "../api/auth";
import { fetchWishlist } from "../api/wishlist";
import { mask_name } from "../utils/masking";
import {
	format_mmss,
	is_login_locked,
	is_password_expired,
	login_lock_remaining_ms,
	migrate_password_changed_at,
	record_login_failure,
	record_login_success,
	should_require_captcha,
	ensure_password_changed_at_initialized,
} from "../utils/accessControl";

import { SimpleCaptcha } from "./SimpleCaptcha";

type AuthUser = {
	name: string;
	email?: string;
};

function parse_user_id(res: any): string | null {
	const data = res?.data;
	const cand = data?.id ?? data?.userId ?? data?.user_id;
	if (typeof cand === "number" && Number.isFinite(cand)) return String(cand);
	if (typeof cand === "string" && cand.trim()) return cand.trim();
	return null;
}

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

		const name = localStorage.getItem("userName") || localStorage.getItem("name");
		const storedEmail = localStorage.getItem("email") || undefined;

		return { name: name || "사용자", email: storedEmail };
	});

	// 접근통제: 로그인 잠금/캡챠
	const [captchaValid, setCaptchaValid] = useState(true);
	const captchaRequired = useMemo(() => {
		return should_require_captcha(email.trim());
	}, [email]);

	const [lockRemaining, setLockRemaining] = useState(0);

	useEffect(() => {
		const em = email.trim();
		if (!em) {
			setLockRemaining(0);
			return;
		}

		const tick = () => setLockRemaining(login_lock_remaining_ms(em));
		tick();

		const id = window.setInterval(tick, 500);
		return () => window.clearInterval(id);
	}, [email]);

	const locked = useMemo(() => {
		const em = email.trim();
		if (!em) return false;
		return is_login_locked(em);
	}, [email, lockRemaining]);

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

	const onQuickLogin = async (e?: React.FormEvent) => {
		e?.preventDefault();

		setErrorMsg(null);

		const em = email.trim();
		const pw = password.trim();

		if (!em || !pw) {
			setErrorMsg("이메일과 비밀번호를 입력하세요.");
			return;
		}

		if (locked) {
			setErrorMsg(`로그인이 잠겨 있습니다. ${format_mmss(lockRemaining)} 후 다시 시도해 주세요.`);
			return;
		}

		if (captchaRequired && !captchaValid) {
			setErrorMsg("캡챠 인증을 완료해 주세요.");
			return;
		}

		try {
			setSubmitting(true);
			const res = await login(em, pw);

			if (res.status !== "success" || !res.data) {
				const st = record_login_failure(em);
				if (st.lock_until && st.lock_until > Date.now()) {
					setErrorMsg(
						`로그인 실패가 누적되어 계정이 잠겼습니다. ${format_mmss(
							login_lock_remaining_ms(em),
						)} 후 다시 시도해 주세요.`,
					);
					return;
				}
				const remaining = Math.max(0, 5 - st.count);
				setErrorMsg((res.message || "로그인 실패") + ` (남은 시도: ${remaining}회)`);
				return;
			}

			const userId = parse_user_id(res);
			if (!userId) {
				localStorage.removeItem("userId");
				setErrorMsg("로그인 정보 처리 중 문제가 발생했습니다. 다시 시도해주세요.");
				return;
			}

			// 성공 처리
			record_login_success(em);
			localStorage.setItem("userId", userId);
			localStorage.setItem("userName", String(res.data?.name ?? ""));
			localStorage.setItem("email", String(res.data?.email ?? em));

			// 비밀번호 변경일(가입 직후 이메일 기준 저장) → 로그인 후 userId 기준으로 마이그레이션
			migrate_password_changed_at(String(res.data?.email ?? em), userId);
			ensure_password_changed_at_initialized(userId);

			// 홈에서 로그인했을 때는 홈에 그대로 머물며 UI만 로그인 상태로 갱신
			setIsAuthed(true);
			setUser({
				name: String(res.data?.name ?? "사용자"),
				email: String(res.data?.email ?? em),
			});

			// 비밀번호 유효기간 만료 시 프로필로 이동하여 변경 유도
			if (is_password_expired(userId)) {
				navigate("/profile", {
					replace: true,
					state: {
						passwordExpired: true,
						fromAfterChange: "/",
					},
				});
				return;
			}

			navigate("/", { replace: true });
		} catch (err: any) {
			const em2 = email.trim();
			if (em2) {
				const st = record_login_failure(em2);
				if (st.lock_until && st.lock_until > Date.now()) {
					setErrorMsg(
						`로그인 실패가 누적되어 계정이 잠겼습니다. ${format_mmss(
							login_lock_remaining_ms(em2),
						)} 후 다시 시도해 주세요.`,
					);
				} else {
					const remaining = Math.max(0, 5 - st.count);
					setErrorMsg((err?.message || "로그인 중 오류가 발생했습니다.") + ` (남은 시도: ${remaining}회)`);
				}
				return;
			}
			setErrorMsg(err?.message || "로그인 중 오류가 발생했습니다.");
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
				{/* 상단: 홈 전용 컴팩트 소개 */}
				<div className="flex items-end justify-between gap-4 mb-6">
					<div>
						<div className="text-sm text-slate-500">AI 기반 입찰 탐색 · 관리</div>
						<h1 className="text-2xl font-bold tracking-tight text-slate-900">
							공고를 더 빠르게 찾고, 더 확실하게 준비하세요
						</h1>
					</div>
					<div className="hidden md:flex gap-2"></div>
				</div>

				{/* 메인 그리드: 좌(4메뉴 박스) + 우(로그인/회원정보) */}
				<div className="grid grid-cols-12 gap-6">
					{/* LEFT */}
					<div className="col-span-12 lg:col-span-8 space-y-6">
						<section className="bg-white border rounded-2xl p-5 shadow-sm">
							<div className="flex items-center justify-between mb-4">
								<h2 className="font-semibold text-slate-900">바로가기</h2>
								<div className="text-sm text-slate-500">핵심 기능 4개를 빠르게 접근</div>
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
									desc={`관심 공고 관리${wishlistCount ? ` · ${wishlistCount}건` : ""}`}
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

						{/* 빈 느낌 줄이기: 홈에서 보여주는 “요약 카드” */}
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
							</div>

						</section>
					</div>

					{/* RIGHT */}
					{/* 데스크탑에서 우측 박스가 너무 위로 붙어 보이는 현상 방지 */}
					<div className="col-span-12 lg:col-span-4 lg:pt-6">
						{!isAuthed ? (
							<aside className="bg-white border rounded-2xl p-5 shadow-sm">
								<h3 className="text-lg font-semibold text-slate-900 mb-2">로그인</h3>
								<p className="text-sm text-slate-500 mb-4">
									로그인하면 장바구니/알림/AI 기능을 이용할 수 있습니다.
								</p>

								<form className="space-y-3" onSubmit={onQuickLogin}>
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

									<SimpleCaptcha
										required={captchaRequired}
										onValidChange={setCaptchaValid}
									/>

									{errorMsg && <div className="text-sm text-red-600">{errorMsg}</div>}

									{locked && (
										<div className="text-sm text-amber-700">
											로그인 시도가 제한되었습니다. {format_mmss(lockRemaining)} 후 다시
											시도해 주세요.
										</div>
									)}

									<button
										type="submit"
										disabled={submitting || locked || (captchaRequired && !captchaValid)}
										className="w-full h-11 rounded-xl bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-60"
									>
										{submitting ? "로그인 중..." : "로그인"}
									</button>

									<div className="grid grid-cols-1 gap-3">
										<button
											type="button"
											onClick={() => navigate("/signup")}
											className="h-11 rounded-xl border hover:bg-slate-50"
										>
											회원가입
										</button>
									</div>

									<div className="flex justify-between text-sm text-slate-500 pt-2">
										<button
											type="button"
											onClick={() => navigate("/find-account")}
											className="hover:text-blue-600 hover:underline"
										>
											계정 찾기
										</button>
										<button
											type="button"
											onClick={() => navigate("/reset-password")}
											className="hover:text-blue-600 hover:underline"
										>
											비밀번호 찾기
										</button>
									</div>
								</form>
							</aside>
						) : (
							<aside className="bg-white border rounded-2xl p-5 shadow-sm">
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

function MiniStat({ label, value, sub }: { label: string; value: string; sub: string }) {
	return (
		<div className="border rounded-2xl p-4 bg-slate-50">
			<div className="text-sm text-slate-500">{label}</div>
			<div className="text-2xl font-bold text-slate-900">{value}</div>
			<div className="text-sm text-slate-500">{sub}</div>
		</div>
	);
}

function MiniKpi({ label, value }: { label: string; value: string }) {
	return (
		<div className="border rounded-2xl p-4 bg-slate-50">
			<div className="text-sm text-slate-500">{label}</div>
			<div className="text-xl font-bold text-slate-900">{value}</div>
		</div>
	);
}
