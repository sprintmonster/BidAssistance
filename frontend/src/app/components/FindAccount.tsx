import { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./ui/card";
import { Building2 } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

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

interface FindAccountPageProps {
    onFindAccount: (payload: {
        name: string;
        birthDate: string; // YYYY-MM-DD
        questionKey: SecurityQuestionKey;
        answer: string;
    }) => void | Promise<void>;
    onNavigateToLogin: () => void;
}

export function FindAccountPage({ onFindAccount, onNavigateToLogin }: FindAccountPageProps) {
    const [formData, setFormData] = useState({
        name: "",
        birthDate: "",
        questionKey: "" as SecurityQuestionKey | "",
        answer: "",
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!formData.name || !formData.birthDate || !formData.questionKey || !formData.answer) return;

        onFindAccount({
            name: formData.name,
            birthDate: formData.birthDate,
            questionKey: formData.questionKey as SecurityQuestionKey,
            answer: formData.answer,
        });
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
                    <CardTitle className="text-2xl text-center">계정 찾기</CardTitle>
                    <CardDescription className="text-center">
                        가입 시 등록한 정보로 이메일(계정)을 확인합니다
                    </CardDescription>
                </CardHeader>

                <form onSubmit={handleSubmit}>
                    <CardContent className="space-y-4">
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
                                onValueChange={(value) => setFormData({ ...formData, questionKey: value as SecurityQuestionKey })}
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
                    </CardContent>

                    <CardFooter className="flex flex-col space-y-4">
                        <Button type="submit" className="w-full">
                            계정 찾기
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
