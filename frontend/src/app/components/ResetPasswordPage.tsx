import { useMemo, useState } from "react";
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
import { Building2 } from "lucide-react";

// ====== 질문(0~3 고정 매핑) ======
const SECURITY_QUESTIONS = [
    "가장 기억에 남는 선생님 성함은?",
    "첫 반려동물 이름은?",
    "출생한 도시는?",
    "가장 좋아하는 음식은?",
] as const;

interface ResetPasswordPageProps {
    onNavigateToLogin: () => void;
}

export function ResetPasswordPage({ onNavigateToLogin }: ResetPasswordPageProps) {
    // 단계: identify(email+birthDate) -> challenge(question+answer)
    const [step, setStep] = useState<"identify" | "challenge">("identify");

    const [identifiedEmail, setIdentifiedEmail] = useState<string>("");
    const [questionIndex, setQuestionIndex] = useState<number | null>(null);
    const [recoverySessionId, setRecoverySessionId] = useState<string>("");

    const [formData, setFormData] = useState({
        email: "",
        name : "",
        birthDate: "",
        answer: "",
    });

    const [message, setMessage] = useState<{
        type: "error" | "success";
        text: string;
    } | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const canIdentify = useMemo(() => {
        return Boolean(formData.email.trim() && formData.name.trim() && formData.birthDate);
    }, [formData.email, formData.name, formData.birthDate]);


    const canChallenge = useMemo(() => {
        return Boolean(questionIndex !== null && formData.answer);
    }, [questionIndex, formData.answer]);

    // function generateTempPassword(length = 6) {
    //     const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    //     let out = "";
    //
    //     const cryptoObj = (globalThis as unknown as { crypto?: Crypto }).crypto;
    //     if (cryptoObj?.getRandomValues) {
    //         const buf = new Uint32Array(length);
    //         cryptoObj.getRandomValues(buf);
    //         for (let i = 0; i < length; i += 1) {
    //             out += chars[buf[i] % chars.length];
    //         }
    //         return out;
    //     }
    //
    //     for (let i = 0; i < length; i += 1) {
    //         out += chars[Math.floor(Math.random() * chars.length)];
    //     }
    //     return out;
    // }

    const handleIdentify = async (e: React.FormEvent) => {
        e.preventDefault();
        setMessage(null);

        if (!canIdentify) return;

        try {
            setIsSubmitting(true);

            const payload = {
                email: formData.email.trim(),
                name: formData.name.trim(),
                birth: formData.birthDate, // LocalDate: "YYYY-MM-DD"
            };

            const res = await fetch("/api/users/recovery_question", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const json = await res.json().catch(() => null);

            if (!res.ok || json?.status === "error") {
                // 정의서: 401 / 404
                const msg =
                    json?.message ??
                    (res.status === 401
                        ? "본인 확인에 실패했습니다."
                        : res.status === 404
                            ? "가입된 계정을 찾을 수 없습니다."
                            : "요청에 실패했습니다. 다시 시도해 주세요.");
                setMessage({ type: "error", text: msg });
                return;
            }

            // 성공: { recoverySessionId, questionId }
            const sid = json?.data?.recoverySessionId;
            const qid = json?.data?.questionId;

            if (typeof sid !== "string" || typeof qid !== "number") {
                setMessage({ type: "error", text: "서버 응답 형식이 올바르지 않아요." });
                return;
            }

            setRecoverySessionId(sid);
            setQuestionIndex(qid); // questionId를 questionIndex로 그대로 사용 (0~3이면 OK)
            setStep("challenge");

            setMessage({
                type: "success",
                text: "확인 완료. 가입 시 설정한 질문에 답변해 주세요.",
            });
        } catch {
            setMessage({
                type: "error",
                text: "서버에 연결할 수 없어요. 잠시 후 다시 시도해 주세요.",
            });
        } finally {
            setIsSubmitting(false);
        }
    };


    const handleSendTempPassword = async (e: React.FormEvent) => {
        e.preventDefault();
        setMessage(null);

        if (!canChallenge) return;
        if (!recoverySessionId) {
            setMessage({ type: "error", text: "세션 정보가 없어요. 다시 시도해 주세요." });
            setStep("identify");
            return;
        }

        try {
            setIsSubmitting(true);

            const payload = {
                recoverySessionId,
                answer: formData.answer.trim(),
            };

            const res = await fetch("/api/users/reset_password", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const json = await res.json().catch(() => null);

            if (!res.ok || json?.status === "error") {
                // 정의서: 401 / 410 / 429
                const msg =
                    json?.message ??
                    (res.status === 401
                        ? "답변이 일치하지 않습니다."
                        : res.status === 410
                            ? "요청이 만료되었습니다. 다시 시도해 주세요."
                            : res.status === 429
                                ? "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."
                                : "요청에 실패했습니다. 다시 시도해 주세요.");
                setMessage({ type: "error", text: msg });

                // 만료면 identify로 되돌리는게 UX 좋음
                if (res.status === 410) {
                    setStep("identify");
                    setRecoverySessionId("");
                    setQuestionIndex(null);
                    setFormData((prev) => ({ ...prev, answer: "" }));
                }
                return;
            }

            // 성공: { data: { message } }
            const serverMsg =
                json?.data?.message ??
                "임시 비밀번호가 발급되었습니다. 이메일을 확인해 주세요.";

            setMessage({ type: "success", text: serverMsg });

            // 원하면 성공 후 로그인 페이지 이동
            // setTimeout(() => onNavigateToLogin(), 400);
        } catch {
            setMessage({
                type: "error",
                text: "서버에 연결할 수 없어요. 잠시 후 다시 시도해 주세요.",
            });
        } finally {
            setIsSubmitting(false);
        }
    };


    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
            <Card className="w-full max-w-md">
                <CardHeader className="space-y-1">
                    <div className="flex items-center justify-center mb-4">
                        <img
                            src="/logo_mini.png"
                            alt="입찰인사이트 로고(축소판)"
                            className="h-20 w-auto block object-contain"
                        />
                    </div>
                    <CardTitle className="text-2xl text-center">비밀번호 찾기</CardTitle>
                    <CardDescription className="text-center">
                        본인 확인 후 임시 비밀번호를 이메일로 전송합니다
                    </CardDescription>
                </CardHeader>

                <form onSubmit={step === "identify" ? handleIdentify : handleSendTempPassword}>
                    <CardContent className="space-y-4">
                        {message && (
                            <div
                                role="alert"
                                className={[
                                    "rounded-md px-3 py-2 text-sm",
                                    message.type === "success"
                                        ? "bg-green-50 text-green-700"
                                        : "bg-red-50 text-red-700",
                                ].join(" ")}
                            >
                                {message.text}
                            </div>
                        )}

                        {step === "identify" ? (
                            <>
                                <div className="space-y-2">
                                    <Label htmlFor="name">이름</Label>
                                    <Input
                                        id="name"
                                        value={formData.name}
                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                        required
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="email">이메일</Label>
                                    <Input
                                        id="email"
                                        type="email"
                                        placeholder="name@company.com"
                                        value={formData.email}
                                        onChange={(e) =>
                                            setFormData({ ...formData, email: e.target.value })
                                        }
                                        required
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="birthDate">생년월일</Label>
                                    <Input
                                        id="birthDate"
                                        type="date"
                                        value={formData.birthDate}
                                        onChange={(e) =>
                                            setFormData({ ...formData, birthDate: e.target.value })
                                        }
                                        required
                                    />
                                </div>
                            </>
                        ) : (
                            <>
                                <div className="text-sm text-muted-foreground">
                                    계정: <span className="font-medium">{identifiedEmail}</span>
                                </div>

                                <div className="space-y-2">
                                    <Label>가입 시 설정한 질문</Label>
                                    <div className="rounded-md border bg-white px-3 py-2 text-sm">
                                        {questionIndex !== null ? SECURITY_QUESTIONS[questionIndex] : ""}
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="answer">답변</Label>
                                    <Input
                                        id="answer"
                                        value={formData.answer}
                                        onChange={(e) =>
                                            setFormData({ ...formData, answer: e.target.value })
                                        }
                                        required
                                    />
                                </div>

                                <Button
                                    type="button"
                                    variant="outline"
                                    className="w-full"
                                    onClick={() => {
                                        setStep("identify");
                                        setIdentifiedEmail("");
                                        setQuestionIndex(null);
                                        setRecoverySessionId("");
                                        setFormData({
                                            name : formData.name,
                                            email: formData.email,
                                            birthDate: formData.birthDate,
                                            answer: "",
                                        });
                                        setMessage(null);
                                    }}
                                >
                                    다시 입력하기
                                </Button>
                            </>
                        )}
                    </CardContent>

                    <CardFooter className="flex flex-col space-y-4">
                        <Button
                            type="submit"
                            className="w-full"
                            disabled={
                                (step === "identify" && (!canIdentify || isSubmitting)) ||
                                (step === "challenge" && (!canChallenge || isSubmitting))
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

                        <div className="text-sm text-center text-gray-600">
                            로그인 화면으로 돌아가기{" "}
                            <button
                                type="button"
                                onClick={onNavigateToLogin}
                                className="text-blue-600 hover:underline"
                            >
                                로그인
                            </button>
                        </div>
                    </CardFooter>
                </form>
            </Card>
        </div>
    );
}
