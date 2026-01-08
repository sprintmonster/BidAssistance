import { useMemo, useState } from "react";
import type { Page } from "../../types/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Badge } from "./ui/badge";
import { toast } from "sonner";
import {
  LayoutDashboard,
  Search,
  TrendingUp,
  ShoppingCart,
  Bell,
  MessageSquare,
  User,
  LogIn,
  LogOut,
  Sparkles,
} from "lucide-react";

type Props = {
  isAuthenticated: boolean;
  userEmail: string;

  cartCount: number;
  unreadCount: number;

  onNavigate: (page: Page, bidId?: number) => void;
  onLogin: (email: string) => void;
  onNavigateToSignup: () => void;
  onLogout: () => void;
};

export function HomePage({
  isAuthenticated,
  userEmail,
  cartCount,
  unreadCount,
  onNavigate,
  onLogin,
  onNavigateToSignup,
  onLogout,
}: Props) {
  const [email, setEmail] = useState(userEmail || "");
  const [password, setPassword] = useState("");
  const [query, setQuery] = useState("");

  const quickLinks = useMemo(
    () => [
      { id: "dashboard" as Page, label: "대시보드", icon: LayoutDashboard },
      { id: "bids" as Page, label: "공고 찾기", icon: Search },
      { id: "analytics" as Page, label: "낙찰 분석", icon: TrendingUp },
      { id: "cart" as Page, label: "장바구니", icon: ShoppingCart, badge: cartCount },
      { id: "notifications" as Page, label: "알림", icon: Bell, badge: unreadCount },
      { id: "chatbot" as Page, label: "AI 챗봇", icon: MessageSquare },
      { id: "profile" as Page, label: "마이페이지", icon: User },
    ],
    [cartCount, unreadCount]
  );

  const runAISearch = () => {
    if (!query.trim()) {
      toast.info("검색어를 입력해 주세요.");
      return;
    }
    // 1) 당장 결과 UI를 붙이기 전이라면, AI 챗봇으로 보내는 것이 자연스럽습니다.
    localStorage.setItem("chatbot.initialQuery", query.trim());
    onNavigate("chatbot");
  };

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* 중앙 영역 */}
      <div className="col-span-12 lg:col-span-8 space-y-6">
        {/* 퀵 링크 박스들 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {quickLinks.map((x) => {
            const Icon = x.icon;
            return (
              <button
                key={x.id}
                onClick={() => onNavigate(x.id)}
                className="text-left"
              >
                <Card className="hover:shadow-sm transition">
                  <CardContent className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="h-9 w-9 rounded-md bg-gray-100 flex items-center justify-center">
                        <Icon className="h-4 w-4" />
                      </div>
                      <div className="font-medium">{x.label}</div>
                    </div>
                    {"badge" in x && x.badge && x.badge > 0 ? (
                      <Badge>{x.badge}</Badge>
                    ) : null}
                  </CardContent>
                </Card>
              </button>
            );
          })}
        </div>

        {/* AI 검색 패널 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5" />
              AI 기반 공고 검색
            </CardTitle>
            <CardDescription>
              자연어로 입력하면 조건을 해석해 공고 탐색/분석 흐름으로 연결합니다.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>무엇을 찾고 있나요?</Label>
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder='예: "서울/경기 10억~50억 시설공사, 마감 임박 우선"'
              />
            </div>

            <div className="flex gap-2">
              <Button onClick={runAISearch} className="gap-2">
                <Sparkles className="h-4 w-4" />
                AI로 검색
              </Button>
              <Button variant="outline" onClick={() => onNavigate("bids")}>
                공고 리스트로 이동
              </Button>
            </div>

            <div className="text-sm text-muted-foreground">
              현재는 AI 챗봇 화면으로 검색어를 전달합니다. 추후 이 패널에서 바로 결과 리스트/필터를
              렌더링하도록 확장할 수 있습니다.
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 우측 사이드바 */}
      <div className="col-span-12 lg:col-span-4 space-y-6">
        {!isAuthenticated ? (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <LogIn className="h-5 w-5" />
                로그인
              </CardTitle>
              <CardDescription>로그인하면 장바구니/알림/AI 기능을 이용할 수 있습니다.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>이메일</Label>
                <Input value={email} onChange={(e) => setEmail(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>비밀번호</Label>
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>

              <div className="flex gap-2">
                <Button
                  className="flex-1"
                  onClick={() => {
                    if (!email.trim()) {
                      toast.error("이메일을 입력해 주세요.");
                      return;
                    }
                    onLogin(email.trim());
                  }}
                >
                  로그인
                </Button>
                <Button variant="outline" className="flex-1" onClick={onNavigateToSignup}>
                  회원가입
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="h-5 w-5" />
                내 정보
              </CardTitle>
              <CardDescription>계정/회사/알림 설정을 관리할 수 있습니다.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1">
                <div className="font-semibold">김철수</div>
                <div className="text-sm text-muted-foreground">{userEmail}</div>
                <div className="flex gap-2 pt-2">
                  <Badge>호반건설</Badge>
                  <Badge variant="outline">중형 건설사</Badge>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <Button variant="outline" onClick={() => onNavigate("profile")}>
                  마이페이지
                </Button>
                <Button variant="outline" onClick={() => onNavigate("notifications")}>
                  알림
                </Button>
              </div>

              <Button variant="ghost" className="w-full justify-start" onClick={onLogout}>
                <LogOut className="h-4 w-4 mr-2" />
                로그아웃
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}