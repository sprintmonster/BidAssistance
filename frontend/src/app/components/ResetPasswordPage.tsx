import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

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
import { api } from "../api/client";

const SECURITY_QUESTIONS = [
	"가장 기억에 남는 선생님 성함은?",
	"첫 반려동물 이름은?",
	"출생한 도시는?",
	"가장 좋아하는 음식은?",
] as const;

type Notice = { type: "error" | "success"; text: string } | null;

export function ResetPasswordPage() {
	const navigate = useNavigate();

	const [step, setStep] = useState<"identify" | "challenge">("identify");

	const [identifiedEmail, setIdentifiedEmail] = useState<string>("");
	const [questionIndex, setQuestionIndex] = useState<number | null>(null);
	const [recoverySessionId, setRecoverySessionId] = useState<string>("");

	const [formData, setFormData] = useState({
		email: "",
		name: "",
		birthDate: "",
		answer: "",
	});

	const [message, setMessage] = useState<Notice>(null);
	const [isSubmitting, setIsSubmitting] = useState(false);

	const canIdentify = useMemo(() => {
		return Boolean(formData.email.trim() && formData.name.trim() && formData.birthDate);
	}, [formData.email, formData.name, formData.birthDate]);

	const canChallenge = useMemo(() => {
		return Boolean(questionIndex !== null && formData.answer.trim());
	}, [questionIndex, formData.answer]);

    const handleIdentify = async (e: React.FormEvent) => {
        e.preventDefault();
        setMessage(null);
        if (!canIdentify) return;

        try {
            setIsSubmitting(true);

            type RecoveryQuestionRes = {
                status: "success" | "error";
                message?: string;
                data?: { userId: string | number; questionIndex: number };
            };

            const qs = new URLSearchParams({
                email: formData.email.trim(),
                name: formData.name.trim(),
                birth: formData.birthDate, // "YYYY-MM-DD"
            }).toString();

            const json = await api<RecoveryQuestionRes>(`/users/recovery_question?${qs}`, {
                method: "GET",
            });

            if (json.status === "error") {
                setMessage({ type: "error", text: json.message ?? "요청에 실패했습니다." });
                return;
            }

            const uid = json?.data?.userId;
            const qidx = json?.data?.questionIndex;

            // qidx 검증 (0~3만 허용)
            if (typeof qidx !== "number" || qidx < 0 || qidx >= SECURITY_QUESTIONS.length) {
                setMessage({ type: "error", text: "질문 정보를 불러오지 못했어요." });
                return;
            }

            //  recoverySessionId는 백엔드에 없으니 'userId'를 임시로 저장해둠(아래 reset에 쓰진 않음)
            setRecoverySessionId(String(uid ?? ""));
            setQuestionIndex(qidx);
            setIdentifiedEmail(formData.email.trim());
            setStep("challenge");

            setMessage({ type: "success", text: "확인 완료. 가입 시 설정한 질문에 답변해 주세요." });
        } catch (e) {
            setMessage({ type: "error", text: e instanceof Error ? e.message : "요청에 실패했습니다." });
        } finally {
            setIsSubmitting(false);
        }
    };


    const handleSendTempPassword = async (e: React.FormEvent) => {
        e.preventDefault();
        setMessage(null);
        if (!canChallenge) return;

        try {
            setIsSubmitting(true);

            const payload = {
                email: formData.email.trim(),
                name: formData.name.trim(),
                birth: formData.birthDate,
                answer: formData.answer.trim(),
            };

            const json = await api<{
                status: "success" | "error";
                message?: string;
                data?: { message?: string };
            }>("/users/reset_password", {
                method: "POST",
                body: JSON.stringify(payload),
            });

            if (json.status === "error") {
                setMessage({ type: "error", text: json.message ?? "요청에 실패했습니다." });
                return;
            }

            const serverMsg = json?.data?.message ?? "임시 비밀번호가 발급되었습니다. 이메일을 확인해 주세요.";
            setMessage({ type: "success", text: serverMsg });
        } catch (e) {
            setMessage({ type: "error", text: e instanceof Error ? e.message : "요청에 실패했습니다." });
        } finally {
            setIsSubmitting(false);
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

				<form onSubmit={step === "identify" ? handleIdentify : handleSendTempPassword}>
					<CardContent className="space-y-5">
						<div className="space-y-1">
							<CardTitle className="text-2xl text-center">비밀번호 찾기</CardTitle>
							<CardDescription className="text-center">
								본인 확인 후 임시 비밀번호를 이메일로 전송합니다
							</CardDescription>
						</div>

						{message && (
							<div
								role="alert"
								className={[
									"rounded-lg border px-3 py-2 text-sm",
									message.type === "success"
										? "border-green-200 bg-green-50 text-green-700"
										: "border-red-200 bg-red-50 text-red-700",
								].join(" ")}
							>
								{message.text}
							</div>
						)}

						{step === "identify" ? (
							<>
								<div className="space-y-2">
									<Label htmlFor="name" className="text-sm font-medium">이름</Label>
									<Input
										id="name"
										value={formData.name}
										onChange={(e) => setFormData({ ...formData, name: e.target.value })}
										className="h-11"
										required
									/>
								</div>

								<div className="space-y-2">
									<Label htmlFor="email" className="text-sm font-medium">이메일</Label>
									<Input
										id="email"
										type="email"
										placeholder="name@company.com"
										value={formData.email}
										onChange={(e) => setFormData({ ...formData, email: e.target.value })}
										className="h-11"
										autoComplete="username"
										required
									/>
								</div>

								<div className="space-y-2">
									<Label htmlFor="birthDate" className="text-sm font-medium">생년월일</Label>
									<Input
										id="birthDate"
										type="date"
										value={formData.birthDate}
										onChange={(e) => setFormData({ ...formData, birthDate: e.target.value })}
										className="h-11"
										required
									/>
								</div>
							</>
						) : (
							<>
								<div className="text-sm text-muted-foreground">
									계정: <span className="font-medium text-slate-900">{identifiedEmail}</span>
								</div>

								<div className="space-y-2">
									<Label className="text-sm font-medium">가입 시 설정한 질문</Label>
									<div className="rounded-lg border bg-white px-3 py-3 text-sm leading-relaxed">
										{questionIndex !== null ? SECURITY_QUESTIONS[questionIndex] : ""}
									</div>
								</div>

								<div className="space-y-2">
									<Label htmlFor="answer" className="text-sm font-medium">답변</Label>
									<Input
										id="answer"
										value={formData.answer}
										onChange={(e) => setFormData({ ...formData, answer: e.target.value })}
										className="h-11"
										required
									/>
								</div>

								<Button
									type="button"
									variant="outline"
									className="w-full h-11"
									onClick={() => {
										setStep("identify");
										setIdentifiedEmail("");
										setQuestionIndex(null);
										setRecoverySessionId("");
										setFormData((prev) => ({ ...prev, answer: "" }));
										setMessage(null);
									}}
									disabled={isSubmitting}
								>
									다시 입력하기
								</Button>
							</>
						)}
					</CardContent>

					<CardFooter className="flex flex-col gap-3 pt-6">
						<Button
							type="submit"
							className="w-full h-11 text-base font-semibold"
							disabled={
								isSubmitting ||
								(step === "identify" && !canIdentify) ||
								(step === "challenge" && !canChallenge)
							}
						>
							{step === "identify"
								? isSubmitting
									? "확인 중..."
									: "다음"
								: isSubmitting
									? "전송 중..."
									: "임시 비밀번호 전송"}
						</Button>

						<div className="flex justify-between gap-4 text-sm">
							<button
								type="button"
								onClick={() => navigate("/login")}
								className="text-muted-foreground hover:text-slate-900 hover:underline"
							>
								로그인
							</button>
							<button
								type="button"
								onClick={() => navigate("/find-account")}
								className="text-muted-foreground hover:text-slate-900 hover:underline"
							>
								계정 찾기
							</button>
						</div>
					</CardFooter>
				</form>
			</Card>
		</div>
	);
}
