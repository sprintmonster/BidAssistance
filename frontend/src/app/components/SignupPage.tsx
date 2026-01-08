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
import { Building2 } from "lucide-react";
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
}

// ==============================
// ✅ [추가] 로컬 "DB"(localStorage) 유틸
// ==============================
type LocalUser = {
    email: string;
    password: string; // ⚠️ 데모/로컬용: 실제 서비스에서는 서버 + 해시/솔트 필수
    companyName: string;
    companyType: string;
    name: string;
    createdAt: string;
    consents: {
        privacyRequired: boolean;
        marketingOptional: boolean;
    };
    // ✅ [추가] 계정 찾기용 질문/답변 저장
    recoveryQA: {
        question: string;
        answer: string; // ⚠️ 데모/로컬용: 실제 서비스에서는 서버측에서 안전하게 처리 권장
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

function findUserByEmail(email: string): LocalUser | undefined {
    const users = readUsers();
    const normalized = email.trim().toLowerCase();
    return users.find((u) => u.email.trim().toLowerCase() === normalized);
}

function upsertUser(user: LocalUser) {
    const users = readUsers();
    const normalized = user.email.trim().toLowerCase();
    const idx = users.findIndex(
        (u) => u.email.trim().toLowerCase() === normalized
    );
    if (idx >= 0) users[idx] = user;
    else users.push(user);
    writeUsers(users);
}

export function SignupPage({ onSignup, onNavigateToLogin }: SignupPageProps) {
    const [formData, setFormData] = useState({
        email: "",
        password: "",
        confirmPassword: "",
        companyName: "",
        companyType: "",
        name: "",
        // ✅ [추가] 계정 찾기 질문/답변 입력값
        recoveryQuestion: "",
        recoveryAnswer: "",
    });

    // ✅ [추가] 질문 preset (원하면 더 추가 가능)
    const recoveryQuestionOptions = useMemo(
        () => [
            "기억에 남는 첫 직장 이름은?",
            "가장 좋아하는 음식은?",
            "어릴 때 살던 동네(시/구)는?",
            "가장 좋아하는 영화는?",
        ],
        []
    );

    // ==============================
    // ✅ [추가] 개인정보 동의 상태
    // ==============================
    const [consents, setConsents] = useState({
        privacyRequired: false,
        marketingOptional: false,
    });

    // ✅ [추가] 에러/상태 메시지
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // ✅ [수정] 버튼 활성 조건에 질문/답변 필수 포함
    const canSubmit = useMemo(() => {
        if (!formData.name.trim()) return false;
        if (!formData.email.trim()) return false;
        if (!formData.companyName.trim()) return false;
        if (!formData.companyType.trim()) return false;
        if (!formData.password) return false;
        if (formData.password !== formData.confirmPassword) return false;
        if (!consents.privacyRequired) return false;

        // ✅ [추가] 계정 찾기 질문/답변 필수
        if (!formData.recoveryQuestion.trim()) return false;
        if (!formData.recoveryAnswer.trim()) return false;

        return true;
    }, [formData, consents.privacyRequired]);

    // ✅ [추가] 입력 변경 시 메시지 초기화
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
            // ✅ [추가] 필수 동의 체크
            if (!consents.privacyRequired) {
                setError("개인정보 수집·이용(필수)에 동의해야 가입할 수 있어요.");
                return;
            }

            // ✅ [추가] 비밀번호 확인
            if (formData.password !== formData.confirmPassword) {
                setError("비밀번호와 비밀번호 확인이 일치하지 않아요.");
                return;
            }

            // ✅ [추가] 회사 유형 선택 확인
            if (!formData.companyType) {
                setError("회사 유형을 선택해 주세요.");
                return;
            }

            // ✅ [추가] 계정 찾기 질문/답변 검증
            if (!formData.recoveryQuestion.trim()) {
                setError("계정 찾기 질문을 선택해 주세요.");
                return;
            }
            if (!formData.recoveryAnswer.trim()) {
                setError("계정 찾기 답변을 입력해 주세요.");
                return;
            }

            // ✅ [추가] 이메일 중복 검사
            const existing = findUserByEmail(formData.email);
            if (existing) {
                setError("이미 가입된 이메일이에요. 로그인해 주세요.");
                return;
            }

            // ✅ [수정] 로컬 DB 저장 데이터에 recoveryQA 포함
            const newUser: LocalUser = {
                email: formData.email.trim(),
                password: formData.password,
                companyName: formData.companyName.trim(),
                companyType: formData.companyType,
                name: formData.name.trim(),
                createdAt: new Date().toISOString(),
                consents: {
                    privacyRequired: consents.privacyRequired,
                    marketingOptional: consents.marketingOptional,
                },
                recoveryQA: {
                    question: formData.recoveryQuestion.trim(),
                    answer: formData.recoveryAnswer.trim(),
                },
            };

            upsertUser(newUser);

            setSuccess("가입이 완료됐어요! 로그인 페이지로 이동합니다.");

            onSignup(newUser.email); // 기존 콜백 유지 (추적/토스트 등 용도)

            // ✅ [수정] 가입 성공 시 로그인으로 이동
            setTimeout(() => {
                onNavigateToLogin();
            }, 300);
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
                    <CardTitle className="text-2xl text-center">회원가입</CardTitle>
                    <CardDescription className="text-center">
                        입찰 기회를 놓치지 마세요
                    </CardDescription>
                </CardHeader>

                <form onSubmit={handleSubmit}>
                    <CardContent className="space-y-4">
                        {/* ✅ [추가] 에러/성공 메시지 */}
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
                            <Label htmlFor="companyName">회사명</Label>
                            <Input
                                id="companyName"
                                value={formData.companyName}
                                onChange={(e) =>
                                    setFormData({ ...formData, companyName: e.target.value })
                                }
                                required
                            />
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="companyType">회사 유형</Label>
                            <Select
                                value={formData.companyType}
                                onValueChange={(value) =>
                                    setFormData({ ...formData, companyType: value })
                                }
                            >
                                <SelectTrigger id="companyType">
                                    <SelectValue placeholder="선택하세요" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="small">소형 건설사</SelectItem>
                                    <SelectItem value="medium">중형 건설사</SelectItem>
                                    <SelectItem value="large">대형 건설사</SelectItem>
                                </SelectContent>
                            </Select>
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

                        {/* ==============================
                ✅ [추가] 계정 찾기 질문/답변 섹션
               ============================== */}
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
                                    {recoveryQuestionOptions.map((q) => (
                                        <SelectItem key={q} value={q}>
                                            {q}
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

                        {/* ==============================
                ✅ [추가] 개인정보 동의 섹션
               ============================== */}
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
                            disabled={!canSubmit || isSubmitting} // ✅ [유지/추가] 필수 조건 충족 전 비활성화
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
