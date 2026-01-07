import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend, ScatterChart, Scatter } from "recharts";
import { TrendingUp, Award, Target, AlertTriangle } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Badge } from "./ui/badge";

export function AnalyticsReport() {
  // Mock data for analytics
  const winningBidsByAgency = [
    { agency: "서울시", count: 23, avgBid: 85.2 },
    { agency: "경기도", count: 18, avgBid: 78.5 },
    { agency: "인천시", count: 15, avgBid: 82.1 },
    { agency: "부산시", count: 12, avgBid: 79.8 },
    { agency: "대전시", count: 9, avgBid: 81.3 },
  ];

  const bidPatternsByBudget = [
    { range: "~10억", avgWinRate: 88.5, submissions: 45 },
    { range: "10-30억", avgWinRate: 84.2, submissions: 67 },
    { range: "30-50억", avgWinRate: 82.1, submissions: 38 },
    { range: "50-100억", avgWinRate: 79.8, submissions: 24 },
    { range: "100억+", avgWinRate: 77.3, submissions: 12 },
  ];

  const monthlyWinRate = [
    { month: "7월", winRate: 82, avgBid: 83.5 },
    { month: "8월", winRate: 85, avgBid: 84.2 },
    { month: "9월", winRate: 81, avgBid: 82.8 },
    { month: "10월", winRate: 87, avgBid: 85.1 },
    { month: "11월", winRate: 83, avgBid: 83.9 },
    { month: "12월", winRate: 86, avgBid: 84.7 },
  ];

  const competitorAnalysis = [
    { company: "A건설", wins: 34, avgBid: 83.2, specialty: "도로/토목" },
    { company: "B건설", wins: 28, avgBid: 85.1, specialty: "건축" },
    { company: "C건설", wins: 22, avgBid: 82.5, specialty: "환경" },
    { company: "D건설", wins: 19, avgBid: 84.8, specialty: "건축" },
    { company: "E건설", wins: 15, avgBid: 81.9, specialty: "도로/토목" },
  ];

  const priceDistribution = [
    { estimate: 80, actual: 82.5, outcome: "낙찰" },
    { estimate: 85, actual: 84.1, outcome: "낙찰" },
    { estimate: 88, actual: 87.9, outcome: "낙찰" },
    { estimate: 83, actual: 85.2, outcome: "유찰" },
    { estimate: 86, actual: 83.8, outcome: "낙찰" },
    { estimate: 84, actual: 86.1, outcome: "낙찰" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl mb-2">낙찰 분석 리포트</h2>
        <p className="text-muted-foreground">경쟁사 분석과 낙찰 패턴을 확인하세요</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm">평균 낙찰률</CardTitle>
            <Award className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl">84.2%</div>
            <p className="text-xs text-muted-foreground">최근 6개월</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm">낙찰 건수</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl">186건</div>
            <p className="text-xs text-muted-foreground">전년 대비 +12%</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm">경쟁률</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl">3.8:1</div>
            <p className="text-xs text-muted-foreground">평균 입찰 참가사</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm">유찰률</CardTitle>
            <AlertTriangle className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl">7.2%</div>
            <p className="text-xs text-muted-foreground">업계 평균: 9.5%</p>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analytics */}
      <Tabs defaultValue="agency" className="space-y-4">
        <TabsList>
          <TabsTrigger value="agency">기관별 분석</TabsTrigger>
          <TabsTrigger value="budget">금액대별 패턴</TabsTrigger>
          <TabsTrigger value="trend">낙찰 추이</TabsTrigger>
          <TabsTrigger value="competitors">경쟁사 분석</TabsTrigger>
        </TabsList>

        <TabsContent value="agency" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>발주기관별 낙찰 현황</CardTitle>
              <CardDescription>주요 발주기관의 낙찰 통계 및 평균 낙찰률</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={winningBidsByAgency}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="agency" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="count" fill="#3b82f6" name="낙찰 건수" />
                  <Bar yAxisId="right" dataKey="avgBid" fill="#8b5cf6" name="평균 낙찰률 (%)" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="budget" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>금액대별 낙찰 패턴</CardTitle>
              <CardDescription>예산 규모에 따른 낙찰률 분석</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={bidPatternsByBudget}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="submissions" fill="#10b981" name="참여 건수" />
                  <Bar yAxisId="right" dataKey="avgWinRate" fill="#f59e0b" name="평균 낙찰률 (%)" />
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <p className="text-sm">
                  <strong>인사이트:</strong> 10~30억원 규모의 공사에서 가장 높은 낙찰률(84.2%)을 기록하고 있습니다. 
                  이 구간에 집중하여 입찰 전략을 수립하는 것을 권장합니다.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trend" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>월별 낙찰 추이</CardTitle>
              <CardDescription>낙찰률과 평균 투찰가의 변화</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={monthlyWinRate}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="winRate" stroke="#3b82f6" strokeWidth={2} name="낙찰률 (%)" />
                  <Line type="monotone" dataKey="avgBid" stroke="#ec4899" strokeWidth={2} name="평균 투찰률 (%)" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="competitors" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>주요 경쟁사 분석</CardTitle>
              <CardDescription>경쟁사별 낙찰 실적 및 특화 분야</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {competitorAnalysis.map((competitor, index) => (
                  <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <span className="font-semibold">{competitor.company}</span>
                        <Badge variant="outline">{competitor.specialty}</Badge>
                      </div>
                      <div className="mt-2 flex gap-6 text-sm text-muted-foreground">
                        <span>낙찰 {competitor.wins}건</span>
                        <span>평균 낙찰률 {competitor.avgBid}%</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-blue-600">#{index + 1}</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
