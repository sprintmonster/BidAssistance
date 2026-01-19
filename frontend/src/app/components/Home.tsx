import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api/auth";
import { fetchWishlist } from "../api/wishlist";

type AuthUser = {
  name: string;
  email?: string;
};

export function Home() {
  const navigate = useNavigate();

  const isAuthed = useMemo(
    () => !!localStorage.getItem("accessToken"),
    []
  );

  const [wishlistCount, setWishlistCount] = useState(0);

  // 홈 우측 로그인 폼
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const user: AuthUser | null = useMemo(() => {
    const name = localStorage.getItem("name");
    const storedEmail = localStorage.getItem("email") || undefined;
    if (!localStorage.getItem("accessToken")) return null;
    return { name: name || "사용자", email: storedEmail };
  }, []);

  useEffect(() => {
    if (!localStorage.getItem("accessToken")) return;
    fetchWishlist()
      .then((items) => setWishlistCount(items.length))
      .catch(() => setWishlistCount(0));
  }, []);

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

      localStorage.setItem("accessToken", res.data.accessToken);
      localStorage.setItem("refreshToken", res.data.refreshToken);
      localStorage.setItem("userId", res.data.userId);
      localStorage.setItem("name", res.data.name);
      localStorage.setItem("email", email.trim());

      navigate("/dashboard");
    } catch (e: any) {
      setErrorMsg(e?.message || "로그인 중 오류가 발생했습니다.");
    } finally {
      setSubmitting(false);
    }
  };

  const onLogout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("refreshToken");
    localStorage.removeItem("userId");
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
          <div className="hidden md:flex gap-2">
          </div>
        </div>

        {/* 메인 그리드: 좌(4메뉴 박스) + 우(로그인/회원정보) */}
        <div className="grid grid-cols-12 gap-6">
          {/* LEFT */}
          <div className="col-span-12 lg:col-span-8 space-y-6">
            <section className="bg-white border rounded-2xl p-5 shadow-sm">
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
                <MiniStat label="관심 공고" value={String(wishlistCount)} sub="장바구니" />
              </div>

              <div className="mt-4 text-sm text-slate-500">
                상단 AI 검색창에서 자연어로 조건을 입력하면 공고 탐색 흐름으로 바로 연결됩니다.
              </div>
            </section>
          </div>

          {/* RIGHT */}
          <div className="col-span-12 lg:col-span-4">
            {!localStorage.getItem("accessToken") ? (
              <aside className="bg-white border rounded-2xl p-5 shadow-sm">
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

                  <div className="flex justify-between text-sm text-slate-500 pt-2">
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
                </div>
              </aside>
            ) : (
              <aside className="bg-white border rounded-2xl p-5 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <div className="text-sm text-slate-500">환영합니다</div>
                    <div className="text-lg font-semibold text-slate-900">
                      {user?.name ?? "사용자"}
                    </div>
                    {user?.email && (
                      <div className="text-sm text-slate-500">{user.email}</div>
                    )}
                  </div>
                  <div className="w-11 h-11 rounded-xl bg-blue-600 text-white flex items-center justify-center font-bold">
                    {user?.name?.slice(0, 1) ?? "U"}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <MiniKpi label="장바구니" value={String(wishlistCount)} />
                  <MiniKpi label="알림" value="0" />
                </div>

                <div className="space-y-2">
                  <button
                    onClick={() => navigate("/dashboard")}
                    className="w-full h-11 rounded-xl bg-slate-900 text-white hover:bg-slate-800"
                  >
                    대시보드로 이동
                  </button>
                  <button
                    onClick={() => navigate("/cart")}
                    className="w-full h-11 rounded-xl border hover:bg-slate-50"
                  >
                    장바구니 보기
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
