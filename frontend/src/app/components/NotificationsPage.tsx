import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Bell, AlertCircle, FileText, RefreshCw, XCircle, Settings } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";

interface NotificationsPageProps {
  onNavigate: (page: string) => void;
}

export function NotificationsPage({ onNavigate }: NotificationsPageProps) {
  const notifications = [
    {
      id: 1,
      type: "deadline",
      title: "마감 임박 알림",
      message: "서울시 강남구 도로 보수공사가 2일 후 마감됩니다",
      time: "2시간 전",
      read: false,
      urgent: true,
    },
    {
      id: 2,
      type: "correction",
      title: "정정공고 발표",
      message: "경기도 성남시 공공건물 신축공사의 예산이 87억원에서 92억원으로 변경되었습니다",
      time: "5시간 전",
      read: false,
      urgent: false,
    },
    {
      id: 3,
      type: "reannouncement",
      title: "재공고 등록",
      message: "인천 항만시설 보수공사가 재공고 되었습니다",
      time: "1일 전",
      read: true,
      urgent: false,
    },
    {
      id: 4,
      type: "unsuccessful",
      title: "유찰 공고",
      message: "부산시 해운대구 주차장 건설이 유찰되었습니다. 재입찰 예정",
      time: "1일 전",
      read: true,
      urgent: false,
    },
    {
      id: 5,
      type: "new",
      title: "신규 공고",
      message: "관심 지역(서울)에 새로운 공고 3건이 등록되었습니다",
      time: "2일 전",
      read: true,
      urgent: false,
    },
    {
      id: 6,
      type: "deadline",
      title: "마감 임박 알림",
      message: "인천광역시 연수구 학교시설 개선공사가 4일 후 마감됩니다",
      time: "2일 전",
      read: true,
      urgent: false,
    },
    {
      id: 7,
      type: "correction",
      title: "정정공고 발표",
      message: "대전시 유성구 복지센터 리모델링의 기술요건이 변경되었습니다",
      time: "3일 전",
      read: true,
      urgent: false,
    },
  ];

  const getIcon = (type: string) => {
    switch (type) {
      case "deadline":
        return <AlertCircle className="h-5 w-5 text-orange-600" />;
      case "correction":
        return <FileText className="h-5 w-5 text-blue-600" />;
      case "reannouncement":
        return <RefreshCw className="h-5 w-5 text-purple-600" />;
      case "unsuccessful":
        return <XCircle className="h-5 w-5 text-red-600" />;
      default:
        return <Bell className="h-5 w-5 text-green-600" />;
    }
  };

  const getTypeLabel = (type: string) => {
    switch (type) {
      case "deadline":
        return "마감임박";
      case "correction":
        return "정정공고";
      case "reannouncement":
        return "재공고";
      case "unsuccessful":
        return "유찰";
      default:
        return "신규공고";
    }
  };

  const unreadCount = notifications.filter(n => !n.read).length;
  const urgentNotifications = notifications.filter(n => n.urgent);
  const allNotifications = notifications;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl mb-2">알림 센터</h2>
          <p className="text-muted-foreground">중요한 입찰 정보를 놓치지 마세요</p>
        </div>
        <Button variant="outline" onClick={() => onNavigate("profile")}>
          <Settings className="h-4 w-4 mr-2" />
          알림 설정
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">읽지 않은 알림</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{unreadCount}개</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">긴급 알림</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{urgentNotifications.length}개</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">전체 알림</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{allNotifications.length}개</div>
          </CardContent>
        </Card>
      </div>

      {/* Notifications List */}
      <Tabs defaultValue="all" className="space-y-4">
        <TabsList>
          <TabsTrigger value="all">
            전체 ({allNotifications.length})
          </TabsTrigger>
          <TabsTrigger value="unread">
            읽지 않음 ({unreadCount})
          </TabsTrigger>
          <TabsTrigger value="urgent">
            긴급 ({urgentNotifications.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-3">
          {allNotifications.map((notification) => (
            <Card
              key={notification.id}
              className={notification.read ? "bg-gray-50" : "border-blue-200 bg-blue-50"}
            >
              <CardContent className="pt-6">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0">{getIcon(notification.type)}</div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant={notification.urgent ? "destructive" : "secondary"}>
                        {getTypeLabel(notification.type)}
                      </Badge>
                      {!notification.read && <Badge variant="default">새 알림</Badge>}
                    </div>
                    <h4 className="font-semibold mb-1">{notification.title}</h4>
                    <p className="text-sm text-muted-foreground mb-2">{notification.message}</p>
                    <p className="text-xs text-muted-foreground">{notification.time}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="unread" className="space-y-3">
          {notifications
            .filter(n => !n.read)
            .map((notification) => (
              <Card key={notification.id} className="border-blue-200 bg-blue-50">
                <CardContent className="pt-6">
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0">{getIcon(notification.type)}</div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant={notification.urgent ? "destructive" : "secondary"}>
                          {getTypeLabel(notification.type)}
                        </Badge>
                        <Badge variant="default">새 알림</Badge>
                      </div>
                      <h4 className="font-semibold mb-1">{notification.title}</h4>
                      <p className="text-sm text-muted-foreground mb-2">{notification.message}</p>
                      <p className="text-xs text-muted-foreground">{notification.time}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
        </TabsContent>

        <TabsContent value="urgent" className="space-y-3">
          {urgentNotifications.map((notification) => (
            <Card key={notification.id} className="border-red-200 bg-red-50">
              <CardContent className="pt-6">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0">{getIcon(notification.type)}</div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant="destructive">{getTypeLabel(notification.type)}</Badge>
                      {!notification.read && <Badge variant="default">새 알림</Badge>}
                    </div>
                    <h4 className="font-semibold mb-1">{notification.title}</h4>
                    <p className="text-sm text-muted-foreground mb-2">{notification.message}</p>
                    <p className="text-xs text-muted-foreground">{notification.time}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>
      </Tabs>

      {/* Notification Settings Preview */}
      <Card>
        <CardHeader>
          <CardTitle>알림 설정</CardTitle>
          <CardDescription>받고 싶은 알림 유형을 선택하세요</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>마감 임박 알림</Label>
              <p className="text-sm text-muted-foreground">마감 3일 전부터 알림</p>
            </div>
            <Switch defaultChecked />
          </div>
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>정정공고 알림</Label>
              <p className="text-sm text-muted-foreground">공고 내용 변경 시 알림</p>
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
              <p className="text-sm text-muted-foreground">관심 지역 신규 공고 알림</p>
            </div>
            <Switch />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
