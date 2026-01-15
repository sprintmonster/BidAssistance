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

// legacy(키 문자열 저장) → 숫자 인덱스로 변환(과거 데이터 호환용)
const LEGACY_KEY_TO_INDEX: Record<string, number> = {
    favorite_teacher: 0,
    first_pet: 1,
    birth_city: 2,
    favorite_food: 3,
};

function resolveQuestionIndex(recoveryQA: {
    questionIndex?: number;
    question?: string;
}): number | null {
    const idx = recoveryQA?.questionIndex;
    if (Number.isInteger(idx) && idx >= 0 && idx < SECURITY_QUESTIONS.length) {
        return idx;
    }

    const legacy = (recoveryQA?.question ?? "").trim();
    if (legacy && legacy in LEGACY_KEY_TO_INDEX) {
        return LEGACY_KEY_TO_INDEX[legacy];
    }
    return null;
}

// ====== localStorage user shape (SignupPage랑 맞춤) ======
type LocalUser = {
    email: string;
    password: string;
    companyName: string;
    name: string;
    birthDate: string; // YYYY-MM-DD
    createdAt: string;
    consents: {
        privacyRequired: boolean;
        marketingOptional: boolean;
    };
    recoveryQA: {
        questionIndex?: number; // 0~3
        question?: string; // legacy
        answer: string;
    };
};

const LS_USERS_KEY = "bidassistance_users_v1";

function readUsers(): LocalUser[] {
    try {
        const raw = localStorage.getItem(LS_USERS_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? (parsed as LocalUser[]) : [];
    } catch {
        return [];
    }
}

function writeUsers(users: LocalUser[]) {
    localStorage.setItem(LS_USERS_KEY, JSON.stringify(users));
}

function normalize(s: string) {
    return s.trim().toLowerCase();
}

interface ResetPasswordPageProps {
    onNavigateToLogin: () => void;
}

export function ResetPasswordPage({ onNavigateToLogin }: ResetPasswordPageProps) {
    // 단계: identify(email+birthDate) -> challenge(question+answer)
    const [step, setStep] = useState<"identify" | "challenge">("identify");

    const [identifiedEmail, setIdentifiedEmail] = useState<string>("");
    const [questionIndex, setQuestionIndex] = useState<number | null>(null);

    const [formData, setFormData] = useState({
        email: "",
        birthDate: "",
        answer: "",
    });

    const [message, setMessage] = useState<{
        type: "error" | "success";
        text: string;
    } | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const canIdentify = useMemo(() => {
        return Boolean(formData.email && formData.birthDate);
    }, [formData.email, formData.birthDate]);

    const canChallenge = useMemo(() => {
        return Boolean(questionIndex !== null && formData.answer);
    }, [questionIndex, formData.answer]);

    function generateTempPassword(length = 6) {
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let out = "";

        const cryptoObj = (globalThis as unknown as { crypto?: Crypto }).crypto;
        if (cryptoObj?.getRandomValues) {
            const buf = new Uint32Array(length);
            cryptoObj.getRandomValues(buf);
            for (let i = 0; i < length; i += 1) {
                out += chars[buf[i] % chars.length];
            }
            return out;
        }

        for (let i = 0; i < length; i += 1) {
            out += chars[Math.floor(Math.random() * chars.length)];
        }
        return out;
    }

    const handleIdentify = async (e: React.FormEvent) => {
        e.preventDefault();
        setMessage(null);

        if (!canIdentify) return;

        try {
            setIsSubmitting(true);

            const users = readUsers();
            const target = users.find(
                (u) => normalize(u.email) === normalize(formData.email)
            );

            if (!target) {
                setMessage({
                    type: "error",
                    text: "해당 이메일로 가입된 계정을 찾을 수 없어요.",
                });
                return;
            }

            if (target.birthDate !== formData.birthDate) {
                setMessage({ type: "error", text: "생년월일이 일치하지 않아요." });
                return;
            }

            const qIndex = resolveQuestionIndex(target.recoveryQA);
            if (qIndex === null) {
                setMessage({
                    type: "error",
                    text: "계정에 비밀번호 찾기 질문이 설정되어 있지 않아요. 고객센터에 문의해 주세요.",
                });
                return;
            }

            setIdentifiedEmail(target.email);
            setQuestionIndex(qIndex);
            setStep("challenge");
            setMessage({
                type: "success",
                text: "확인 완료. 가입 시 설정한 질문에 답변해 주세요.",
            });
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

            const users = readUsers();
            const idx = users.findIndex(
                (u) => normalize(u.email) === normalize(identifiedEmail)
            );

            if (idx < 0) {
                setMessage({
                    type: "error",
                    text: "계정을 찾을 수 없어요. 다시 시도해 주세요.",
                });
                setStep("identify");
                return;
            }

            const target = users[idx];

            if (target.birthDate !== formData.birthDate) {
                setMessage({
                    type: "error",
                    text: "정보가 변경되었어요. 다시 시도해 주세요.",
                });
                setStep("identify");
                return;
            }

            const storedQIndex = resolveQuestionIndex(target.recoveryQA);
            if (storedQIndex === null || storedQIndex !== questionIndex) {
                setMessage({
                    type: "error",
                    text: "질문 정보가 일치하지 않아요. 다시 시도해 주세요.",
                });
                setStep("identify");
                return;
            }

            if (normalize(target.recoveryQA?.answer ?? "") !== normalize(formData.answer)) {
                setMessage({ type: "error", text: "질문 답변이 일치하지 않아요." });
                return;
            }

            // ✅ 임시 비밀번호 생성 (6자리, 영어+숫자)
            const tempPassword = generateTempPassword(6);

            // ✅ (데모/로컬) 임시 비밀번호로 교체
            // 실제 서비스에서는 서버에서 비밀번호를 재설정하고 이메일 발송까지 처리해야 합니다.
            users[idx] = {
                ...target,
                password: tempPassword,
            };
            writeUsers(users);

            // ✅ 실제 이메일 전송은 백엔드에서 처리해야 합니다.
            // 프론트에서는 성공 메시지만 보여주고, DEV 모드에서만 임시 비밀번호를 노출(테스트 편의).
            const devHint = import.meta.env.DEV
                ? ` (DEV: 임시 비밀번호: ${tempPassword})`
                : "";

            setMessage({
                type: "success",
                text: `임시 비밀번호가 이메일로 전송되었습니다. 메일을 확인해 주세요.${devHint}`,
            });

            setTimeout(() => onNavigateToLogin(), 400);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
            <Card className="w-full max-w-md">
                <CardHeader className="space-y-1">
                    <div className="flex items-center justify-center mb-4">
                        <div className="bg-blue-600 p-3 rounded-lg">
                            <Building2 className="w-8 h-8 text-white" />
                        </div>
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
                                        setFormData({
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
