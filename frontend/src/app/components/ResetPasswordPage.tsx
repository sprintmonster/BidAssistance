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
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "./ui/select";

// ====== 질문 ======
type SecurityQuestionKey =
    | "favorite_teacher"
    | "first_pet"
    | "birth_city"
    | "favorite_food";

const QUESTION_LABELS: Record<SecurityQuestionKey, string> = {
    favorite_teacher: "가장 기억에 남는 선생님 성함은?",
    first_pet: "첫 반려동물 이름은?",
    birth_city: "출생한 도시는?",
    favorite_food: "가장 좋아하는 음식은?",
};

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
        question: string; // key 저장(예: "first_pet")
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
    // 단계: verify -> setPassword
    const [step, setStep] = useState<"verify" | "setPassword">("verify");
    const [verifiedEmail, setVerifiedEmail] = useState<string>("");

    const [formData, setFormData] = useState({
        email: "",
        name: "",
        birthDate: "",
        questionKey: "" as SecurityQuestionKey | "",
        answer: "",
    });

    const [pwData, setPwData] = useState({
        newPassword: "",
        confirmNewPassword: "",
    });

    const [message, setMessage] = useState<{ type: "error" | "success"; text: string } | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const canVerify = useMemo(() => {
        return Boolean(
            formData.email &&
            formData.name &&
            formData.birthDate &&
            formData.questionKey &&
            formData.answer
        );
    }, [formData]);

    const canSetPassword = useMemo(() => {
        if (!pwData.newPassword) return false;
        if (pwData.newPassword !== pwData.confirmNewPassword) return false;
        if (pwData.newPassword.length < 8) return false;
        return true;
    }, [pwData]);

    const passwordGuideMessage = useMemo(() => {
        if (!pwData.newPassword && !pwData.confirmNewPassword) {
            return "새 비밀번호를 입력해 주세요.";
        }

        if (pwData.newPassword.length > 0 && pwData.newPassword.length < 8) {
            return "비밀번호는 8자 이상이어야 합니다.";
        }

        if (
            pwData.newPassword &&
            pwData.confirmNewPassword &&
            pwData.newPassword !== pwData.confirmNewPassword
        ) {
            return "비밀번호와 비밀번호 확인이 일치하지 않습니다.";
        }

        return null; // 모든 조건 통과
    }, [pwData]);


    const handleVerify = async (e: React.FormEvent) => {
        e.preventDefault();
        setMessage(null);

        if (!canVerify) return;

        try {
            setIsSubmitting(true);

            const users = readUsers();
            const target = users.find((u) => normalize(u.email) === normalize(formData.email));

            if (!target) {
                setMessage({ type: "error", text: "해당 이메일로 가입된 계정을 찾을 수 없어요." });
                return;
            }

            // 이름/생년월일 비교
            if (normalize(target.name) !== normalize(formData.name)) {
                setMessage({ type: "error", text: "이름이 일치하지 않아요." });
                return;
            }
            if (target.birthDate !== formData.birthDate) {
                setMessage({ type: "error", text: "생년월일이 일치하지 않아요." });
                return;
            }

            // 질문/답변 비교 (Signup에서 question은 key 형태로 저장했음)
            if (target.recoveryQA?.question !== formData.questionKey) {
                setMessage({ type: "error", text: "선택한 질문이 가입 정보와 일치하지 않아요." });
                return;
            }
            if (normalize(target.recoveryQA?.answer ?? "") !== normalize(formData.answer)) {
                setMessage({ type: "error", text: "질문 답변이 일치하지 않아요." });
                return;
            }

            // 검증 성공 → 비밀번호 설정 단계로 이동
            setVerifiedEmail(target.email);
            setStep("setPassword");
            setMessage({ type: "success", text: "본인 확인이 완료됐어요. 새 비밀번호를 설정해 주세요." });
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleSetPassword = async (e: React.FormEvent) => {
        e.preventDefault();
        setMessage(null);

        if (!canSetPassword) {
            setMessage({ type: "error", text: "새 비밀번호를 확인해 주세요. (8자 이상, 두 칸 일치)" });
            return;
        }

        try {
            setIsSubmitting(true);

            const users = readUsers();
            const idx = users.findIndex((u) => normalize(u.email) === normalize(verifiedEmail));

            if (idx < 0) {
                setMessage({ type: "error", text: "계정을 찾을 수 없어요. 다시 시도해 주세요." });
                setStep("verify");
                return;
            }

            users[idx] = {
                ...users[idx],
                password: pwData.newPassword, // ✅ 여기서 실제로 변경
            };

            writeUsers(users);

            setMessage({ type: "success", text: "비밀번호가 변경되었습니다. 로그인해 주세요." });

            // 바로 로그인 페이지로 보내고 싶으면:
            setTimeout(() => onNavigateToLogin(), 300);
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
                    <CardTitle className="text-2xl text-center">비밀번호 재설정</CardTitle>
                    <CardDescription className="text-center">
                        본인 확인 후 새 비밀번호를 설정합니다
                    </CardDescription>
                </CardHeader>

                <form onSubmit={step === "verify" ? handleVerify : handleSetPassword}>
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

                        {step === "verify" ? (
                            <>
                                <div className="space-y-2">
                                    <Label htmlFor="email">이메일</Label>
                                    <Input
                                        id="email"
                                        type="email"
                                        placeholder="name@company.com"
                                        value={formData.email}
                                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                        required
                                    />
                                </div>

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
                                    <Label htmlFor="birthDate">생년월일</Label>
                                    <Input
                                        id="birthDate"
                                        type="date"
                                        value={formData.birthDate}
                                        onChange={(e) => setFormData({ ...formData, birthDate: e.target.value })}
                                        required
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="question">가입할 때 설정한 질문</Label>
                                    <Select
                                        value={formData.questionKey}
                                        onValueChange={(value) =>
                                            setFormData({ ...formData, questionKey: value as SecurityQuestionKey })
                                        }
                                    >
                                        <SelectTrigger id="question">
                                            <SelectValue placeholder="선택하세요" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {Object.entries(QUESTION_LABELS).map(([key, label]) => (
                                                <SelectItem key={key} value={key}>
                                                    {label}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="answer">답변</Label>
                                    <Input
                                        id="answer"
                                        value={formData.answer}
                                        onChange={(e) => setFormData({ ...formData, answer: e.target.value })}
                                        required
                                    />
                                </div>
                            </>
                        ) : (
                            <>
                                <div className="text-sm text-muted-foreground">
                                    계정: <span className="font-medium">{verifiedEmail}</span>
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="newPassword">새 비밀번호</Label>
                                    <Input
                                        id="newPassword"
                                        type="password"
                                        value={pwData.newPassword}
                                        onChange={(e) => setPwData({ ...pwData, newPassword: e.target.value })}
                                        required
                                    />
                                    <div className="text-xs text-gray-500">8자 이상 권장</div>
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="confirmNewPassword">새 비밀번호 확인</Label>
                                    <Input
                                        id="confirmNewPassword"
                                        type="password"
                                        value={pwData.confirmNewPassword}
                                        onChange={(e) => setPwData({ ...pwData, confirmNewPassword: e.target.value })}
                                        required
                                    />
                                </div>

                                <Button
                                    type="button"
                                    variant="outline"
                                    className="w-full"
                                    onClick={() => {
                                        setStep("verify");
                                        setVerifiedEmail("");
                                        setPwData({ newPassword: "", confirmNewPassword: "" });
                                        setMessage(null);
                                    }}
                                >
                                    다시 본인 확인하기
                                </Button>
                            </>
                        )}
                    </CardContent>

                    <CardFooter className="flex flex-col space-y-4">
                        <Button
                            type="submit"
                            className="w-full"
                            disabled={(step === "verify" && (!canVerify || isSubmitting)) || (step === "setPassword" && (!canSetPassword || isSubmitting))}
                        >
                            {step === "verify"
                                ? isSubmitting
                                    ? "확인 중..."
                                    : "본인 확인"
                                : isSubmitting
                                    ? "변경 중..."
                                    : "비밀번호 변경"}
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
