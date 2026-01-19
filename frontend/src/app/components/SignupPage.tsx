import { useEffect, useMemo, useState } from "react";
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
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "./ui/select";

interface SignupPageProps {
    onSignup: (email: string) => void;
    onNavigateToLogin: () => void;
    onNavigateToHome: () => void;
}

// ====== 질문(0~3 고정 매핑) ======
const SECURITY_QUESTIONS = [
    "가장 기억에 남는 선생님 성함은?",
    "첫 반려동물 이름은?",
    "출생한 도시는?",
    "가장 좋아하는 음식은?",
] as const;


export function SignupPage({ onSignup, onNavigateToLogin, onNavigateToHome, }: SignupPageProps) {
    const [formData, setFormData] = useState({
        email: "",
        password: "",
        confirmPassword: "",
        nickName: "",
        birthDate: "",
        name: "",
        // 숫자 인덱스를 문자열로 들고 있다가 저장 시 Number()로 변환
        recoveryQuestion: "", // "0" | "1" | "2" | "3"
        recoveryAnswer: "",
    });

    // ==============================
    //  개인정보 동의 상태
    // ==============================
    const [consents, setConsents] = useState({
        privacyRequired: false,
        marketingOptional: false,
    });

    //  에러/상태 메시지
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const canSubmit = useMemo(() => {
        if (!formData.name.trim()) return false;
        if (!formData.email.trim()) return false;
        if (!formData.nickName.trim()) return false;
        if (!formData.password) return false;
        if (formData.password !== formData.confirmPassword) return false;
        if (!consents.privacyRequired) return false;

        if (!formData.birthDate) return false;

        if (!formData.recoveryQuestion.trim()) return false;
        if (!formData.recoveryAnswer.trim()) return false;

        return true;
    }, [formData, consents.privacyRequired]);

    useEffect(() => {
        setError(null);
        setSuccess(null);
    }, [formData, consents]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        setIsSubmitting(true);
        setError(null);
        setSuccess(null);

        try {
            if (!consents.privacyRequired) {
                setError("개인정보 수집·이용(필수)에 동의해야 가입할 수 있어요.");
                return;
            }

            if (formData.password !== formData.confirmPassword) {
                setError("비밀번호와 비밀번호 확인이 일치하지 않아요.");
                return;
            }

            if (!formData.birthDate) {
                setError("생년월일을 입력해 주세요.");
                return;
            }

            if (!formData.recoveryQuestion.trim()) {
                setError("계정 찾기 질문을 선택해 주세요.");
                return;
            }
            if (!formData.recoveryAnswer.trim()) {
                setError("계정 찾기 답변을 입력해 주세요.");
                return;
            }


            const questionIndex = Number(formData.recoveryQuestion);
            if (!Number.isInteger(questionIndex) || questionIndex < 0 || questionIndex > 3) {
                setError("계정 찾기 질문을 다시 선택해 주세요.");
                return;
            }

// API 정의서 필드명에 맞춤
            const payload = {
                email: formData.email.trim(),
                password: formData.password,
                name: formData.name.trim(),
                nickname: formData.nickName.trim(), // nickName -> nickname
                role: 0, //  기본: 일반 유저 (00을 int로 쓰면 보통 0)
                question: questionIndex, //  int
                answer: formData.recoveryAnswer.trim(),
                birth: formData.birthDate, //  birthDate -> birth (YYYY-MM-DD)
                tag: 0, // 기본 태그 (규칙 확정되면 바꿔)
            };

            try {
                const res = await fetch("/api/users", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });

                const json = await res.json().catch(() => null);

                if (!res.ok) {
                    // 정의서: 401, 500
                    const msg =
                        json?.message ??
                        (res.status === 401
                            ? "이메일 또는 비밀번호가 올바르지 않습니다."
                            : res.status === 500
                                ? "서버 내부 오류가 발생했습니다. 관리자에게 문의하세요."
                                : "가입에 실패했어요. 입력값을 확인해 주세요.");
                    setError(msg);
                    return;
                }

                // 성공: code 200 { data: { userId, email, nickname } }
                setSuccess("가입이 완료됐어요! 로그인 페이지로 이동합니다.");

                // 필요 시 상위로 전달
                // onSignup(json?.data?.email ?? payload.email);

                setTimeout(() => onNavigateToLogin(), 300);
            } catch {
                setError("서버에 연결할 수 없어요. 잠시 후 다시 시도해 주세요.");
                return;
            }

        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
            <Card className="w-full max-w-md">
                <CardHeader className="space-y-1">
                    <div className="flex items-center justify-center mb-4">
                        <img
                            src="/logo_mini.png"
                            alt="입찰인사이트 로고(축소판)"
                            className="h-20 w-auto block object-contain cursor-pointer hover:opacity-80 hover:scale-105 transition-all duration-200"
                            onClick={onNavigateToHome}
                            title="홈페이지 이동하기"  // 툴팁 추가
                        />
                    </div>
                    <CardTitle className="text-2xl text-center">회원가입</CardTitle>
                    <CardDescription className="text-center">
                        입찰 기회를 놓치지 마세요
                    </CardDescription>
                </CardHeader>

                <form onSubmit={handleSubmit}>
                    <CardContent className="space-y-4">
                        {error && (
                            <div className="text-sm rounded-md bg-red-50 text-red-700 px-3 py-2">
                                {error}
                            </div>
                        )}
                        {success && (
                            <div className="text-sm rounded-md bg-green-50 text-green-700 px-3 py-2">
                                {success}
                            </div>
                        )}

                        <div className="space-y-2">
                            <Label htmlFor="name">이름</Label>
                            <Input
                                id="name"
                                value={formData.name}
                                onChange={(e) =>
                                    setFormData({ ...formData, name: e.target.value })
                                }
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
                            <Label htmlFor="nickName">닉네임</Label>
                            <Input
                                id="nickName"
                                value={formData.nickName}
                                onChange={(e) =>
                                    setFormData({ ...formData, nickName: e.target.value })
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

                        <div className="space-y-2">
                            <Label htmlFor="password">비밀번호</Label>
                            <Input
                                id="password"
                                type="password"
                                value={formData.password}
                                onChange={(e) =>
                                    setFormData({ ...formData, password: e.target.value })
                                }
                                required
                            />
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="confirmPassword">비밀번호 확인</Label>
                            <Input
                                id="confirmPassword"
                                type="password"
                                value={formData.confirmPassword}
                                onChange={(e) =>
                                    setFormData({
                                        ...formData,
                                        confirmPassword: e.target.value,
                                    })
                                }
                                required
                            />
                        </div>

                        {/* 계정 찾기 질문/답변 */}
                        <div className="space-y-2 pt-2">
                            <Label htmlFor="recoveryQuestion">계정 찾기 질문</Label>
                            <Select
                                value={formData.recoveryQuestion}
                                onValueChange={(value) =>
                                    setFormData({ ...formData, recoveryQuestion: value })
                                }
                            >
                                <SelectTrigger id="recoveryQuestion">
                                    <SelectValue placeholder="질문을 선택하세요" />
                                </SelectTrigger>
                                <SelectContent>
                                    {SECURITY_QUESTIONS.map((label, idx) => (
                                        <SelectItem key={String(idx)} value={String(idx)}>
                                            {label}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="recoveryAnswer">계정 찾기 답변</Label>
                            <Input
                                id="recoveryAnswer"
                                value={formData.recoveryAnswer}
                                onChange={(e) =>
                                    setFormData({ ...formData, recoveryAnswer: e.target.value })
                                }
                                placeholder="답변을 입력하세요"
                                required
                            />
                            <div className="text-xs text-gray-500">
                                나중에 계정 찾기/복구에 사용됩니다.
                            </div>
                        </div>

                        {/* 약관 동의 */}
                        <div className="space-y-3 pt-2">
                            <div className="text-sm font-medium">약관 동의</div>

                            <label className="flex items-start gap-2 text-sm text-gray-700">
                                <input
                                    type="checkbox"
                                    className="mt-1"
                                    checked={consents.privacyRequired}
                                    onChange={(e) =>
                                        setConsents({
                                            ...consents,
                                            privacyRequired: e.target.checked,
                                        })
                                    }
                                />
                                <span>
                                    <span className="font-medium">
                                        개인정보 수집·이용 동의(필수)
                                    </span>
                                    <span className="text-red-600"> *</span>
                                    <div className="text-xs text-gray-500 mt-1">
                                        회원가입 및 서비스 제공을 위해 필요합니다.
                                    </div>
                                </span>
                            </label>

                            <label className="flex items-start gap-2 text-sm text-gray-700">
                                <input
                                    type="checkbox"
                                    className="mt-1"
                                    checked={consents.marketingOptional}
                                    onChange={(e) =>
                                        setConsents({
                                            ...consents,
                                            marketingOptional: e.target.checked,
                                        })
                                    }
                                />
                                <span>
                                    마케팅 정보 수신 동의(선택)
                                    <div className="text-xs text-gray-500 mt-1">
                                        이벤트/혜택 알림을 받아볼 수 있어요.
                                    </div>
                                </span>
                            </label>
                        </div>
                    </CardContent>

                    <CardFooter className="flex flex-col space-y-4">
                        <Button
                            type="submit"
                            className="w-full"
                            disabled={!canSubmit || isSubmitting}
                        >
                            {isSubmitting ? "가입 처리 중..." : "가입하기"}
                        </Button>

                        <div className="text-sm text-center text-gray-600">
                            이미 계정이 있으신가요?{" "}
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
