import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { login } from "../api/auth";

import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
	Card,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from "./ui/card";

export function LoginPage() {
	const navigate = useNavigate();
	const location = useLocation() as any;

	// 로그인 성공 후 기본 이동 경로는 "홈(/)".
	// (특정 페이지에서 로그인 페이지로 유도된 경우, state.from을 우선 사용)
	const from = location?.state?.from || "/";

	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");

	const [errorMsg, setErrorMsg] = useState<string | null>(null);
	const [submitting, setSubmitting] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setErrorMsg(null);

        try {
            setSubmitting(true);
            const res = await login(email.trim(), password);

            if (res.status !== "success" || !res.data) {
                setErrorMsg(res.message || "이메일 또는 비밀번호가 올바르지 않습니다.");
                return;
            }

            const id = (res as any)?.data?.id;

            if (typeof id === "number" && Number.isFinite(id)) {
                localStorage.setItem("userId", String(id));
            } else {
                localStorage.removeItem("userId");
                setErrorMsg("로그인 정보 처리 중 문제가 발생했습니다. 다시 시도해주세요.");
                return;
            }

            localStorage.setItem("userName", String(res.data.name ?? ""));
            localStorage.setItem("email", String(res.data.email ?? email.trim()));


            navigate(from, { replace: true });
        } catch (e: any) {
            setErrorMsg(e?.message || "서버 내부 오류가 발생했습니다. 관리자에게 문의하세요.");
        } finally {
            setSubmitting(false);
        }
    };


	return (
		<div className="min-h-screen flex items-center justify-center p-4 bg-slate-950 bg-[radial-gradient(1200px_500px_at_50%_-20%,rgba(59,130,246,0.18),transparent),radial-gradient(900px_420px_at_15%_110%,rgba(99,102,241,0.12),transparent)]">
			<Card className="w-full max-w-[420px] rounded-2xl border-slate-200/60 shadow-xl">
				<CardHeader className="space-y-2 pb-5">
					<div className="flex items-center justify-center">
						<img
							src="/logo2.png"
							alt="입찰인사이트 로고"
							className="h-14 w-auto object-contain cursor-pointer hover:opacity-90 transition"
							onClick={() => navigate("/")}
							title="홈페이지 이동하기"
						/>
					</div>
				</CardHeader>

				<form onSubmit={handleSubmit}>
					<CardContent className="space-y-5">
						<div className="space-y-2">
							<Label htmlFor="email" className="text-sm font-medium">
								이메일
							</Label>
							<Input
								id="email"
								type="email"
								placeholder="name@company.com"
								value={email}
								onChange={(e) => setEmail(e.target.value)}
								className="h-11"
								autoComplete="username"
								required
							/>
						</div>

						<div className="space-y-2">
							<Label htmlFor="password" className="text-sm font-medium">
								비밀번호
							</Label>
							<Input
								id="password"
								type="password"
								value={password}
								onChange={(e) => setPassword(e.target.value)}
								className="h-11"
								autoComplete="current-password"
								required
							/>
						</div>

						{errorMsg && (
							<div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
								{errorMsg}
							</div>
						)}
					</CardContent>

					<CardFooter className="flex flex-col gap-3 pt-0">
						<Button
							type="submit"
							className="w-full h-11 rounded-xl bg-slate-900 text-white hover:bg-slate-800"
							disabled={submitting}
						>
							{submitting ? "로그인 중..." : "로그인"}
						</Button>

						<div className="w-full flex items-center justify-between text-sm text-slate-500">
							<button
								type="button"
								onClick={() => navigate("/")}
								className="hover:text-blue-600 hover:underline"
							>
								홈으로
							</button>
							<div className="flex gap-3">
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
						</div>
					</CardFooter>
				</form>
			</Card>
		</div>
	);
}
