import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Switch } from "./ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Separator } from "./ui/separator";
import { Badge } from "./ui/badge";

interface ProfilePageProps {
  userEmail: string;
}

type ProfileTab = "info" | "company" | "notifications" | "subscription";

const PROFILE_TAB_STORAGE_KEY = "profile.activeTab";

function readInitialTab(): ProfileTab {
  const v = localStorage.getItem(PROFILE_TAB_STORAGE_KEY);
  if (v === "info" || v === "company" || v === "notifications" || v === "subscription") return v;
  return "info";
}

export function ProfilePage({ userEmail }: ProfilePageProps) {
  const [activeTab, setActiveTab] = useState<ProfileTab>(() => readInitialTab());

  useEffect(() => {
    localStorage.setItem(PROFILE_TAB_STORAGE_KEY, activeTab);
  }, [activeTab]);

  // “검은 박스” 탭 스타일
  const tabTriggerClass = useMemo(() => {
    return [
      "px-4 py-2 rounded-md",
      "data-[state=active]:bg-black data-[state=active]:text-white",
      "data-[state=active]:shadow-sm",
      "data-[state=inactive]:bg-transparent data-[state=inactive]:text-muted-foreground",
      "hover:text-foreground",
    ].join(" ");
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl mb-2">마이페이지</h2>
        <p className="text-muted-foreground">계정 정보 및 설정을 관리하세요</p>
      </div>

      {/* Profile Header */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-6">
            <Avatar className="h-20 w-20">
              <AvatarFallback className="bg-blue-600 text-white text-2xl">
                {userEmail.charAt(0).toUpperCase()}
              </AvatarFallback>
            </Avatar>
            <div className="flex-1">
              <h3 className="text-2xl font-bold mb-1">김철수</h3>
              <p className="text-muted-foreground mb-2">{userEmail}</p>
              <div className="flex gap-2">
                <Badge>호반건설</Badge>
                <Badge variant="outline">중형 건설사</Badge>
              </div>
            </div>
            {/*<Button variant="outline">프로필 수정</Button>*/}
          </div>
        </CardContent>
      </Card>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as ProfileTab)} className="space-y-4">
        <TabsList className="bg-transparent p-0 gap-2">
          <TabsTrigger value="info" className={tabTriggerClass}>
            계정 정보
          </TabsTrigger>
          <TabsTrigger value="company" className={tabTriggerClass}>
            회사 정보
          </TabsTrigger>
          <TabsTrigger value="notifications" className={tabTriggerClass}>
            알림 설정
          </TabsTrigger>
          <TabsTrigger value="subscription" className={tabTriggerClass}>
            구독 관리
          </TabsTrigger>
        </TabsList>

        <TabsContent value="info" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>개인 정보</CardTitle>
              <CardDescription>계정의 기본 정보를 관리합니다</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="name">이름</Label>
                  <Input id="name" defaultValue="김철수" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">이메일</Label>
                  <Input id="email" type="email" defaultValue={userEmail} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="phone">연락처</Label>
                  <Input id="phone" defaultValue="010-1234-5678" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="position">직위</Label>
                  <Input id="position" defaultValue="입찰 담당자" />
                </div>
              </div>
              <Separator />
              <div className="space-y-4">
                <h4 className="font-semibold">비밀번호 변경</h4>
                <div className="space-y-2">
                  <Label htmlFor="currentPassword">현재 비밀번호</Label>
                  <Input id="currentPassword" type="password" />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="newPassword">새 비밀번호</Label>
                    <Input id="newPassword" type="password" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirmPassword">비밀번호 확인</Label>
                    <Input id="confirmPassword" type="password" />
                  </div>
                </div>
              </div>
              <div className="flex justify-end">
                <Button>변경사항 저장</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="company" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>회사 정보</CardTitle>
              <CardDescription>소속 회사의 정보를 관리합니다</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="companyName">회사명</Label>
                  <Input id="companyName" defaultValue="호반건설" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="companyType">회사 유형</Label>
                  <Select defaultValue="medium">
                    <SelectTrigger id="companyType">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="small">소형 건설사</SelectItem>
                      <SelectItem value="medium">중형 건설사</SelectItem>
                      <SelectItem value="large">대형 건설사</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="businessNumber">사업자등록번호</Label>
                  <Input id="businessNumber" defaultValue="123-45-67890" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="representative">대표자명</Label>
                  <Input id="representative" defaultValue="홍길동" />
                </div>
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="address">주소</Label>
                  <Input id="address" defaultValue="서울특별시 강남구 테헤란로 123" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="companyPhone">대표 전화</Label>
                  <Input id="companyPhone" defaultValue="02-1234-5678" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="website">웹사이트</Label>
                  <Input id="website" defaultValue="www.hoban.co.kr" />
                </div>
              </div>
              <div className="flex justify-end">
                <Button>변경사항 저장</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>알림 설정</CardTitle>
              <CardDescription>받고 싶은 알림을 선택하세요</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h4 className="font-semibold mb-4">이메일 알림</h4>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>마감 임박 알림</Label>
                      <p className="text-sm text-muted-foreground">입찰 마감 3일 전부터 매일 알림</p>
                    </div>
                    <Switch defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>정정공고 알림</Label>
                      <p className="text-sm text-muted-foreground">관심 공고의 내용이 변경되면 즉시 알림</p>
                    </div>
                    <Switch defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>재공고 알림</Label>
                      <p className="text-sm text-muted-foreground">유찰 후 재공고 시 알림</p>
                    </div>
                    <Switch defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>신규 공고 알림</Label>
                      <p className="text-sm text-muted-foreground">관심 지역에 신규 공고 등록 시 알림</p>
                    </div>
                    <Switch />
                  </div>
                </div>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-4">관심 지역</h4>
                <div className="space-y-2">
                  <Label>알림 받을 지역 선택</Label>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="default">서울</Badge>
                    <Badge variant="default">경기</Badge>
                    <Badge variant="outline" className="cursor-pointer">
                      인천
                    </Badge>
                    <Badge variant="outline" className="cursor-pointer">
                      부산
                    </Badge>
                    <Badge variant="outline" className="cursor-pointer">
                      대전
                    </Badge>
                    <Badge variant="outline" className="cursor-pointer">
                      광주
                    </Badge>
                  </div>
                </div>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-4">예산 범위</h4>
                <div className="space-y-2">
                  <Label>관심있는 예산 규모</Label>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="default">10억 이하</Badge>
                    <Badge variant="default">10-30억</Badge>
                    <Badge variant="outline" className="cursor-pointer">
                      30-50억
                    </Badge>
                    <Badge variant="outline" className="cursor-pointer">
                      50-100억
                    </Badge>
                    <Badge variant="outline" className="cursor-pointer">
                      100억 이상
                    </Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="subscription" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>구독 플랜</CardTitle>
              <CardDescription>현재 이용 중인 플랜 정보</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg mb-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-2xl font-bold mb-1">프로 플랜</h3>
                    <p className="text-muted-foreground">중소형 건설사를 위한 완벽한 솔루션</p>
                  </div>
                  <Badge className="bg-blue-600">활성</Badge>
                </div>
                <div className="flex items-baseline gap-2 mb-4">
                  <span className="text-4xl font-bold">₩89,000</span>
                  <span className="text-muted-foreground">/월</span>
                </div>
                <p className="text-sm text-muted-foreground">다음 결제일: 2026-02-06</p>
              </div>

              <div className="flex gap-3">
                <Button variant="outline" className="flex-1">
                  플랜 변경
                </Button>
                <Button variant="outline" className="flex-1">
                  결제 내역
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
