import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchBids } from "../api/bids";
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
import { ENABLE_TEST_LOGIN, TEST_LOGIN } from "../utils/testLogin";
import { mark_reco_popup_trigger } from "./RecommendedBidsModal";

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
	const [newBidsToday, setNewBidsToday] = useState(0);
	const [closingSoon3Days, setClosingSoon3Days] = useState(0);

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

	const companyLabel = useMemo(() => {
		const n = localStorage.getItem("companyName")?.trim() || "";
		const p = localStorage.getItem("companyPosition")?.trim() || "";
		if (!n && !p) return "";
		if (n && p) return `${n} · ${p}`;
		return n || p;
	}, []);

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

	useEffect(() => {
		const loadHomeKpi = async () => {
			try {
				const bids = await fetchBids();

				const list = Array.isArray(bids)
					? bids
					: Array.isArray((bids as any)?.data)
						? (bids as any).data
						: Array.isArray((bids as any)?.data?.items)
							? (bids as any).data.items
							: [];

				const now = new Date();
				const yyyy = now.getFullYear();
				const mm = now.getMonth();
				const dd = now.getDate();

				const isSameDay = (d: Date) =>
					d.getFullYear() === yyyy && d.getMonth() === mm && d.getDate() === dd;

				const newToday = list.filter((b: any) => {
					const s = String(b.bidStart ?? b.startDate ?? "");
					const d = s ? new Date(s) : null;
					return d && Number.isFinite(d.getTime()) && isSameDay(d);
				}).length;

				const soon3 = list.filter((b: any) => {
					const s = String(b.bidEnd ?? b.endDate ?? "");
					const d = s ? new Date(s) : null;
					if (!d || !Number.isFinite(d.getTime())) return false;

					const diffMs = d.getTime() - now.getTime();
					const diffDays = diffMs / (1000 * 60 * 60 * 24);
					return diffDays >= 0 && diffDays <= 3;
				}).length;

				setNewBidsToday(newToday);
				setClosingSoon3Days(soon3);
			} catch {
				setNewBidsToday(0);
				setClosingSoon3Days(0);
			}
		};

		void loadHomeKpi();
	}, []);

	const onLogout = () => {
		localStorage.removeItem("userId");
		localStorage.removeItem("userName");
		localStorage.removeItem("name");
		localStorage.removeItem("email");
		localStorage.removeItem("role");
		localStorage.removeItem("companyName");
		localStorage.removeItem("companyPosition");
		localStorage.removeItem("accessToken");
		localStorage.removeItem("refreshToken");
		setIsAuthed(false);
		setUser(null);
		setWishlistCount(0);
		navigate("/");
	};

	const doLogin = async (em: string, pw: string) => {
		setErrorMsg(null);

		const emailTrim = em.trim();

		if (!emailTrim || !pw) {
			setErrorMsg("이메일과 비밀번호를 입력해 주세요.");
			return;
		}

		if (is_login_locked(emailTrim)) {
			setErrorMsg(
				`로그인이 잠겨 있습니다. ${format_mmss(login_lock_remaining_ms(emailTrim))} 후 다시 시도해 주세요.`,
			);
			return;
		}

		if (should_require_captcha(emailTrim) && !captchaValid) {
			setErrorMsg("캡챠 인증을 완료해 주세요.");
			return;
		}

		try {
			setSubmitting(true);
			const res = await login(emailTrim, pw);

			if (res.status !== "success" || !res.data) {
				record_login_failure(emailTrim);
				setErrorMsg(res.message || "로그인에 실패했습니다.");
				return;
			}

			const uid = parse_user_id(res);
			if (!uid) {
				record_login_failure(emailTrim);
				setErrorMsg("로그인 응답에 사용자 식별자가 없습니다.");
				return;
			}

			record_login_success(emailTrim);
			ensure_password_changed_at_initialized(emailTrim);

			if (is_password_expired(emailTrim)) {
				setErrorMsg("비밀번호가 만료되었습니다. 비밀번호를 재설정해 주세요.");
				navigate("/reset-password");
				return;
			}

			migrate_password_changed_at(emailTrim);

			localStorage.setItem("userId", uid);
			if (res.data?.name) localStorage.setItem("userName", String(res.data.name));
			if (res.data?.email) localStorage.setItem("email", String(res.data.email));
			if (typeof res.data?.role === "number") localStorage.setItem("role", String(res.data.role));

			if ((res.data as any)?.companyName)
				localStorage.setItem("companyName", String((res.data as any).companyName));
			if ((res.data as any)?.companyPosition)
				localStorage.setItem("companyPosition", String((res.data as any).companyPosition));

			if ((res.data as any)?.accessToken)
				localStorage.setItem("accessToken", String((res.data as any).accessToken));
			if ((res.data as any)?.refreshToken)
				localStorage.setItem("refreshToken", String((res.data as any).refreshToken));

			setIsAuthed(true);
			setUser({
				name: String(res.data?.name ?? "사용자"),
				email: res.data?.email ? String(res.data.email) : undefined,
			});

			mark_reco_popup_trigger();
			navigate("/dashboard");
		} catch (e) {
			record_login_failure(emailTrim);
			setErrorMsg(e instanceof Error ? e.message : "로그인에 실패했습니다.");
		} finally {
			setSubmitting(false);
		}
	};

	const canTestLogin = ENABLE_TEST_LOGIN;

	const onTestLogin = async () => {
		if (!canTestLogin) return;
		await doLogin(TEST_LOGIN.email, TEST_LOGIN.password);
	};

	return (
		<div className="min-h-[calc(100vh-64px)] bg-slate-50">
			<div className="max-w-6xl mx-auto px-4 py-8">
				<div className="grid grid-cols-12 gap-6">
					<div className="col-span-12 lg:col-span-8 space-y-6">
						<section className="bg-white border rounded-2xl p-5 shadow-sm">
							<div className="flex items-center justify-between mb-4">
								<h2 className="text-lg font-semibold text-slate-900">바로가기</h2>
							</div>

							<div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
								<QuickBox
									title="공고 찾기"
									desc="관심있는 공고를 검색하고 필터링합니다."
									onClick={() => navigate("/bids")}
								/>
								<QuickBox
									title="장바구니"
									desc="찜한 공고와 진행 단계를 관리합니다."
									onClick={() => navigate("/cart")}
									badge={wishlistCount}
								/>
								<QuickBox
									title="커뮤니티"
									desc="정보를 공유하고 질문을 남겨보세요."
									onClick={() => navigate("/community")}
								/>
								<QuickBox
									title="알림"
									desc="장바구니 기반 알림을 확인합니다."
									onClick={() => navigate("/notifications")}
								/>
							</div>
						</section>
					</div>

					<div className="col-span-12 lg:col-span-4">
						{!isAuthed ? (
							<aside className="bg-white border rounded-2xl p-5 shadow-sm h-full flex flex-col">
								<div className="mb-4">
									<h3 className="text-lg font-semibold text-slate-900">로그인</h3>
									<div className="text-sm text-slate-500">서비스를 이용하려면 로그인하세요.</div>
								</div>

								<form
									className="space-y-3"
									onSubmit={(e) => {
										e.preventDefault();
										if (submitting) return;
										if (locked) return;
										void doLogin(email, password);
									}}
								>
									<div>
										<label className="text-sm text-slate-600">이메일</label>
										<input
											value={email}
											onChange={(e) => setEmail(e.target.value)}
											className="mt-1 w-full h-11 border rounded-xl px-3 focus:outline-none focus:ring-2 focus:ring-blue-200"
											placeholder="example@email.com"
											autoComplete="email"
										/>
									</div>

									<div>
										<label className="text-sm text-slate-600">비밀번호</label>
										<input
											type="password"
											value={password}
											onChange={(e) => setPassword(e.target.value)}
											className="mt-1 w-full h-11 border rounded-xl px-3 focus:outline-none focus:ring-2 focus:ring-blue-200"
											placeholder="비밀번호"
											autoComplete="current-password"
										/>
									</div>

									{captchaRequired ? (
										<div className="pt-1">
											<SimpleCaptcha onValidated={setCaptchaValid} />
										</div>
									) : null}

									{locked ? (
										<div className="text-xs text-red-600">
											로그인이 잠겨 있습니다. {format_mmss(lockRemaining)} 후 다시 시도해 주세요.
										</div>
									) : null}

									{errorMsg ? <div className="text-sm text-red-600">{errorMsg}</div> : null}

									<button
										type="submit"
										disabled={submitting || (captchaRequired && !captchaValid) || locked}
										className="w-full h-11 rounded-xl bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-50"
									>
										{submitting ? "로그인 중..." : "로그인"}
									</button>

									{canTestLogin ? (
										<button
											type="button"
											onClick={onTestLogin}
											disabled={submitting}
											className="w-full h-11 rounded-xl border hover:bg-slate-50 text-slate-700 disabled:opacity-50"
										>
											테스트 로그인
										</button>
									) : null}

									<div className="flex items-center justify-between text-sm text-slate-600 pt-1">
										<button
											type="button"
											onClick={() => navigate("/signup")}
											className="hover:text-blue-600 hover:underline"
										>
											회원가입
										</button>
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

								<div className="flex-1" />
							</aside>
						) : (
							<aside className="bg-white border rounded-2xl p-5 shadow-sm h-full flex flex-col">
								<div className="flex items-center justify-between mb-4">
									<div>
										<div className="text-sm text-slate-500">환영합니다</div>
										<div className="text-lg font-semibold text-slate-900">
											{mask_name(user?.name ?? "사용자")}
										</div>
										{companyLabel ? (
											<div className="text-xs text-muted-foreground">{companyLabel}</div>
										) : null}

										{user?.email && <div className="text-sm text-slate-500">{user.email}</div>}
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
										onClick={onLogout}
										className="w-full h-11 rounded-xl border hover:bg-slate-50 text-slate-700"
									>
										로그아웃
									</button>
								</div>

								<div className="flex-1" />
							</aside>
						)}
					</div>

					<section className="col-span-12 lg:col-span-8 bg-white border rounded-2xl p-5 shadow-sm">
						<div className="flex items-center justify-between mb-3">
							<h3 className="font-semibold text-slate-900">오늘의 추천</h3>
							<button className="text-sm text-blue-600 hover:underline" onClick={() => navigate("/bids")}>
								더 보기
							</button>
						</div>

						<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
							<MiniStat
								label="신규 공고"
								value={String(newBidsToday)}
								sub="오늘 시작"
								onClick={() => navigate("/dashboard?focus=new")}
							/>
							<MiniStat
								label="마감 임박"
								value={String(closingSoon3Days)}
								sub="3일 이내"
								onClick={() => navigate("/dashboard?focus=closingSoon")}
							/>
						</div>
					</section>

					<div className="hidden lg:block lg:col-span-4" />
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
	onClick,
}: {
	label: string;
	value: string;
	sub: string;
	onClick?: () => void;
}) {
	if (onClick) {
		return (
			<button
				type="button"
				onClick={onClick}
				className="text-left border rounded-2xl p-4 bg-slate-50 hover:bg-blue-50/40 hover:border-blue-200 transition focus:outline-none focus:ring-2 focus:ring-blue-200"
			>
				<div className="flex items-start justify-between gap-3">
					<div>
						<div className="text-sm text-slate-500">{label}</div>
						<div className="text-2xl font-bold text-slate-900">{value}</div>
						<div className="text-sm text-slate-500">{sub}</div>
					</div>
					<div className="text-blue-600 text-sm mt-1">이동 →</div>
				</div>
			</button>
		);
	}

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
