import { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

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


interface FindAccountPageProps {
    onFindAccount: (payload: {
        name: string;
        birthDate: string; // YYYY-MM-DD
        questionIndex: number;
        answer: string;
    }) => void | Promise<void>;
    onNavigateToLogin: () => void;
    onNavigateToHome: () => void;
}

export function FindAccountPage({ onFindAccount, onNavigateToLogin, onNavigateToHome, }: FindAccountPageProps) {
    const [formData, setFormData] = useState({
        name: "",
        birthDate: "",
        answer: "",
    });

    const [step, setStep] = useState<"identify" | "answer" | "result">("identify");
    const [questionIndex, setQuestionIndex] = useState<number | null>(null);
    const [identifiedEmail, setIdentifiedEmail] = useState<string>(""); // 결과로 보여줄 계정(이메일)
    // const [targetEmail, setTargetEmail] = useState<string>("");
    const [requestId, setRequestId] = useState<string>("");

    const handleIdentify = async (e: React.FormEvent) => {
        e.preventDefault();

        const name = formData.name.trim();
        const birth = formData.birthDate; // YYYY-MM-DD

        if (!name || !birth) return;

        try {
            const qs = new URLSearchParams({ name, birth }).toString();

            const res = await fetch(`/api/users/find_email/identify?${qs}`, {
                method: "GET",
            });

            const json = await res.json().catch(() => null);

            if (!res.ok || json?.status === "error") {
                alert(json?.message ?? "해당 계정이 없습니다.");
                return;
            }

            const rid = json?.data?.requestId;
            const qIndex = json?.data?.questionIndex;

            if (typeof rid !== "string" || typeof qIndex !== "number") {
                alert("서버 응답 형식이 올바르지 않아요.");
                return;
            }

            setRequestId(rid);
            setQuestionIndex(qIndex);
            setStep("answer");
        } catch {
            alert("서버에 연결할 수 없어요. 잠시 후 다시 시도해 주세요.");
        }
    };

    const handleVerifyAnswer = async (e: React.FormEvent) => {
        e.preventDefault();

        const answer = formData.answer.trim();
        if (!requestId || !answer) return;

        try {
            const qs = new URLSearchParams({ requestId, answer }).toString();

            const res = await fetch(`/api/users/find_email/verify?${qs}`, {
                method: "GET",
            });

            const json = await res.json().catch(() => null);

            if (!res.ok || json?.status === "error") {
                alert(json?.message ?? "답변이 일치하지 않습니다.");
                return;
            }

            const email = json?.data?.email;
            if (typeof email !== "string") {
                alert("서버 응답 형식이 올바르지 않아요.");
                return;
            }

            setIdentifiedEmail(email);
            setStep("result");
        } catch {
            alert("서버에 연결할 수 없어요. 잠시 후 다시 시도해 주세요.");
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
                    <CardTitle className="text-2xl text-center">계정 찾기</CardTitle>
                    <CardDescription className="text-center">
                        가입 시 등록한 정보로 이메일(계정)을 확인합니다
                    </CardDescription>
                </CardHeader>

                <form onSubmit={step === "identify" ? handleIdentify : step === "answer" ? handleVerifyAnswer : (e) => e.preventDefault()}>
                    <CardContent className="space-y-4">
                        {/* 1) identify 단계: 이름 + 생년월일 */}
                        {step === "identify" && (
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
                                    <Label htmlFor="birthDate">생년월일</Label>
                                    <Input
                                        id="birthDate"
                                        type="date"
                                        value={formData.birthDate}
                                        onChange={(e) => setFormData({ ...formData, birthDate: e.target.value })}
                                        required
                                    />
                                </div>
                            </>
                        )}

                        {/* 2) answer 단계: 질문 출력 + 답변 입력 + 다시 입력 */}
                        {step === "answer" && (
                            <>
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
                                        onChange={(e) => setFormData({ ...formData, answer: e.target.value })}
                                        required
                                    />
                                </div>

                                <Button
                                    type="button"
                                    variant="outline"
                                    className="w-full"
                                    onClick={() => {
                                        setStep("identify");
                                        setQuestionIndex(null);
                                        setRequestId("");
                                        setIdentifiedEmail("");
                                        setFormData({ ...formData, answer: "" });
                                    }}
                                >
                                    다시 입력하기
                                </Button>
                            </>
                        )}

                        {/* 3) result 단계: 계정(이메일) 공개 */}
                        {step === "result" && (
                            <>
                                <div className="rounded-md bg-green-50 text-green-700 px-3 py-2 text-sm">
                                    계정을 찾았어요!
                                </div>

                                <div className="text-sm">
                                    당신의 계정(이메일): <span className="font-medium">{identifiedEmail}</span>
                                </div>

                                <Button type="button" className="w-full" onClick={onNavigateToLogin}>
                                    로그인하러 가기
                                </Button>

                                <Button
                                    type="button"
                                    variant="outline"
                                    className="w-full"
                                    onClick={() => {
                                        setStep("identify");
                                        setQuestionIndex(null);
                                        setRequestId("");
                                        setIdentifiedEmail("");
                                        setFormData({ name: "", birthDate: "", answer: "" });
                                    }}
                                >
                                    다시 찾기
                                </Button>
                            </>
                        )}
                    </CardContent>


                    <CardFooter className="flex flex-col space-y-4">
                        {step !== "result" && (
                            <Button type="submit" className="w-full">
                                {step === "identify" ? "다음" : "계정 확인"}
                            </Button>
                        )}


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
