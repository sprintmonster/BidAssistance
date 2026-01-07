import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Separator } from "./ui/separator";
import { Building, MapPin, Calendar, DollarSign, FileText, AlertTriangle, CheckCircle2, Clock, TrendingUp, ArrowLeft } from "lucide-react";
import { Progress } from "./ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

interface BidSummaryProps {
  bidId?: number;
  onNavigate: (page: string) => void;
}

export function BidSummary({ bidId, onNavigate }: BidSummaryProps) {
  // Mock bid data
  const bid = {
    id: bidId || 1,
    title: "서울시 강남구 도로 보수공사",
    agency: "서울특별시 강남구청",
    region: "서울",
    budget: "35억 원",
    deadline: "2026-01-08",
    announcementDate: "2025-12-15",
    type: "공사",
    status: "진행중",
    description: "강남구 주요 도로의 노후화된 포장 및 배수시설 보수공사",
    requirements: {
      license: ["건설업 면허 (토목공사업)", "ISO 9001 인증"],
      location: "서울특별시 내 본점 또는 지점 소재",
      experience: "최근 3년간 유사공사 실적 2건 이상",
      technicalStaff: "토목기사 2명 이상",
    },
    risks: [
      { level: "high", text: "마감까지 2일 남음 - 서류 준비 시급" },
      { level: "medium", text: "유사 실적 증빙서류 필수" },
      { level: "low", text: "지역 제한 요건 확인 필요" },
    ],
    checklist: [
      { item: "사업자등록증 사본", checked: true },
      { item: "건설업 면허증 사본", checked: true },
      { item: "ISO 9001 인증서", checked: false },
      { item: "유사 실적 증빙서류 (2건)", checked: false },
      { item: "기술자 보유 현황", checked: true },
      { item: "재무제표 (최근 3년)", checked: false },
    ],
    priceGuidance: {
      recommended: "83.5% - 85.2%",
      historical: "최근 유사공사 평균 낙찰률 84.1%",
      competitors: "예상 참여사 4-5개사",
    },
  };

  const completedItems = bid.checklist.filter(item => item.checked).length;
  const completionRate = (completedItems / bid.checklist.length) * 100;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="outline" size="sm" onClick={() => onNavigate("bids")}>
          <ArrowLeft className="h-4 w-4 mr-1" />
          목록으로
        </Button>
      </div>

      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-3">
                <Badge>{bid.type}</Badge>
                <Badge variant="outline">{bid.status}</Badge>
                <Badge variant="destructive">마감임박</Badge>
              </div>
              <CardTitle className="text-2xl mb-2">{bid.title}</CardTitle>
              <CardDescription>{bid.description}</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="flex items-center gap-3">
              <Building className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">발주기관</p>
                <p className="font-semibold">{bid.agency}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <MapPin className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">지역</p>
                <p className="font-semibold">{bid.region}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <DollarSign className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">예산</p>
                <p className="font-semibold">{bid.budget}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Calendar className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">마감일</p>
                <p className="font-semibold text-red-600">{bid.deadline}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Content Tabs */}
      <Tabs defaultValue="summary" className="space-y-4">
        <TabsList>
          <TabsTrigger value="summary">AI 요약</TabsTrigger>
          <TabsTrigger value="checklist">서류 체크리스트</TabsTrigger>
          <TabsTrigger value="risks">리스크 분석</TabsTrigger>
          <TabsTrigger value="price">투찰 가이드</TabsTrigger>
        </TabsList>

        <TabsContent value="summary" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                공고 핵심 요약
              </CardTitle>
              <CardDescription>AI가 분석한 주요 입찰 요건</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h4 className="font-semibold mb-3">📋 자격 요건</h4>
                <ul className="space-y-2">
                  {bid.requirements.license.map((item, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <CheckCircle2 className="h-4 w-4 mt-0.5 text-green-600" />
                      <span className="text-sm">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-3">📍 지역 요건</h4>
                <p className="text-sm">{bid.requirements.location}</p>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-3">📈 실적 요건</h4>
                <p className="text-sm">{bid.requirements.experience}</p>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-3">👷 기술인력 요건</h4>
                <p className="text-sm">{bid.requirements.technicalStaff}</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="checklist" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5" />
                제출서류 체크리스트
              </CardTitle>
              <CardDescription>
                진행률: {completedItems}/{bid.checklist.length} ({completionRate.toFixed(0)}%)
              </CardDescription>
              <Progress value={completionRate} className="mt-2" />
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {bid.checklist.map((item, index) => (
                  <div
                    key={index}
                    className={`flex items-center gap-3 p-3 rounded-lg border ${
                      item.checked ? "bg-green-50 border-green-200" : "bg-gray-50"
                    }`}
                  >
                    {item.checked ? (
                      <CheckCircle2 className="h-5 w-5 text-green-600" />
                    ) : (
                      <Clock className="h-5 w-5 text-gray-400" />
                    )}
                    <span className={item.checked ? "line-through text-muted-foreground" : ""}>
                      {item.item}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="risks" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                리스크 경고
              </CardTitle>
              <CardDescription>참여 전 확인이 필요한 사항</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {bid.risks.map((risk, index) => (
                  <div
                    key={index}
                    className={`flex items-start gap-3 p-4 rounded-lg border ${
                      risk.level === "high"
                        ? "bg-red-50 border-red-200"
                        : risk.level === "medium"
                        ? "bg-yellow-50 border-yellow-200"
                        : "bg-blue-50 border-blue-200"
                    }`}
                  >
                    <AlertTriangle
                      className={`h-5 w-5 mt-0.5 ${
                        risk.level === "high"
                          ? "text-red-600"
                          : risk.level === "medium"
                          ? "text-yellow-600"
                          : "text-blue-600"
                      }`}
                    />
                    <div>
                      <Badge
                        variant={risk.level === "high" ? "destructive" : "outline"}
                        className="mb-2"
                      >
                        {risk.level === "high" ? "높음" : risk.level === "medium" ? "보통" : "낮음"}
                      </Badge>
                      <p className="text-sm">{risk.text}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="price" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                투찰 범위 가이드
              </CardTitle>
              <CardDescription>과거 데이터 기반 추천 투찰률</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
                <p className="text-sm text-muted-foreground mb-2">추천 투찰 범위</p>
                <p className="text-3xl font-bold text-blue-600">{bid.priceGuidance.recommended}</p>
              </div>

              <Separator />

              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">📊 과거 낙찰 데이터</h4>
                  <p className="text-sm text-muted-foreground">{bid.priceGuidance.historical}</p>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">🏢 예상 경쟁 현황</h4>
                  <p className="text-sm text-muted-foreground">{bid.priceGuidance.competitors}</p>
                </div>

                <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <p className="text-sm">
                    <strong>💡 인사이트:</strong> 이 공고와 유사한 규모·지역의 최근 5건 낙찰률은 82.8% ~ 85.9% 범위에 분포합니다. 
                    84.0% 전후의 투찰가가 안정적인 낙찰 가능성을 보입니다.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
