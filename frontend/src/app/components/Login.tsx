import { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./ui/card";
import { Building2 } from "lucide-react";

export interface LoginPageProps {
    onLogin: (email: string) => void;
    onNavigateToSignup: () => void;
    onNavigateToFindAccount?: () => void;
    onNavigateToResetPassword?: () => void;
}

export function LoginPage({
                              onLogin,
                              onNavigateToSignup,
                              onNavigateToFindAccount,
                              onNavigateToResetPassword,
                          }: LoginPageProps) {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (email && password) {
            onLogin(email);
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
                    <CardTitle className="text-2xl text-center">입찰 인텔리전스</CardTitle>
                    <CardDescription className="text-center">
                        건설사를 위한 스마트 입찰 관리 플랫폼
                    </CardDescription>
                </CardHeader>

                <form onSubmit={handleSubmit}>
                    <CardContent className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="email">이메일</Label>
                            <Input
                                id="email"
                                type="email"
                                placeholder="name@company.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="password">비밀번호</Label>
                            <Input
                                id="password"
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                        </div>


                    </CardContent>

                    <CardFooter className="flex flex-col space-y-4">
                        <Button type="submit" className="w-full">
                            로그인
                        </Button>

                        <div className="text-sm text-center text-gray-600">
                            계정이 없으신가요?{" "}
                            <button
                                type="button"
                                onClick={onNavigateToSignup}
                                className="text-blue-600 hover:underline"
                            >
                                회원가입
                            </button>
                        </div>
                        {/* 계정 / 비밀번호 찾기 */}
                        <div className="flex justify-between gap-4 text-sm">
                            <button
                                type="button"
                                onClick={onNavigateToFindAccount}
                                className="text-gray-600 hover:text-blue-600 hover:underline"
                            >
                                계정 찾기
                            </button>
                            <button
                                type="button"
                                onClick={onNavigateToResetPassword}
                                className="text-gray-600 hover:text-blue-600 hover:underline"
                            >
                                비밀번호 찾기
                            </button>
                        </div>

                    </CardFooter>
                </form>
            </Card>
        </div>
    );
}

