import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from "recharts";
import { Bell, FileText, TrendingUp, AlertCircle, Calendar, DollarSign } from "lucide-react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

interface DashboardProps {
  onNavigate: (page: string) => void;
}

export function Dashboard({ onNavigate }: DashboardProps) {
  // Mock data
  const monthlyBids = [
    { month: "7월", count: 45 },
    { month: "8월", count: 52 },
    { month: "9월", count: 48 },
    { month: "10월", count: 61 },
    { month: "11월", count: 58 },
    { month: "12월", count: 67 },
  ];

  const bidsByRegion = [
    { name: "서울", value: 145, color: "#3b82f6" },
    { name: "경기", value: 98, color: "#8b5cf6" },
    { name: "인천", value: 67, color: "#ec4899" },
    { name: "기타", value: 112, color: "#10b981" },
  ];

  const recentAlerts = [
    { id: 1, type: "마감임박", title: "서울시 도로공사 입찰", time: "2시간 전", urgent: true },
    { id: 2, type: "정정공고", title: "경기도 건축공사", time: "5시간 전", urgent: false },
    { id: 3, type: "재공고", title: "인천 항만시설 보수", time: "1일 전", urgent: false },
  ];

  const upcomingDeadlines = [
    { id: 1, title: "서울 강남구 건축공사", deadline: "2026-01-08", amount: "35억" },
    { id: 2, title: "경기 성남시 도로공사", deadline: "2026-01-10", amount: "12억" },
    { id: 3, title: "인천 연수구 시설공사", deadline: "2026-01-12", amount: "8억" },
  ];

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm">신규 공고</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl">67개</div>
            <p className="text-xs text-muted-foreground">이번 달</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm">관심 공고</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl">24개</div>
            <p className="text-xs text-muted-foreground">장바구니</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm">마감 임박</CardTitle>
            <AlertCircle className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl">8개</div>
            <p className="text-xs text-muted-foreground">3일 이내</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm">총 예상액</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl">142억</div>
            <p className="text-xs text-muted-foreground">관심 공고 합계</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>월별 공고 추이</CardTitle>
            <CardDescription>최근 6개월</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={monthlyBids}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>지역별 분포</CardTitle>
            <CardDescription>현재 진행 중인 공고</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={bidsByRegion}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {bidsByRegion.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Alerts and Deadlines */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5" />
              최근 알림
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentAlerts.map((alert) => (
                <div key={alert.id} className="flex items-start justify-between border-b pb-3 last:border-0">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Badge variant={alert.urgent ? "destructive" : "secondary"}>
                        {alert.type}
                      </Badge>
                    </div>
                    <p className="text-sm">{alert.title}</p>
                    <p className="text-xs text-muted-foreground">{alert.time}</p>
                  </div>
                </div>
              ))}
              <Button variant="outline" className="w-full" onClick={() => onNavigate("notifications")}>
                모든 알림 보기
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5" />
              마감 예정
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {upcomingDeadlines.map((item) => (
                <div key={item.id} className="flex items-start justify-between border-b pb-3 last:border-0">
                  <div className="space-y-1">
                    <p className="text-sm">{item.title}</p>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      <span>{item.deadline}</span>
                      <span>예산: {item.amount}</span>
                    </div>
                  </div>
                </div>
              ))}
              <Button variant="outline" className="w-full" onClick={() => onNavigate("bids")}>
                공고 검색하기
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
