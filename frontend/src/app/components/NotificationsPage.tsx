import { useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Bell, AlertCircle, FileText, RefreshCw, XCircle, Settings, CheckCheck } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import type { Page } from "../../types/navigation";

export type NotificationItem = {
  id: number;
  type: string;
  title: string;
  message: string;
  time: string;
  read: boolean;
  urgent: boolean;
};

interface NotificationsPageProps {
  onNavigate: (page: Page, bidId?: number) => void;

  notifications: NotificationItem[];
  onMarkRead: (id: number) => void;
  onMarkAllRead: () => void;

  // true면 알림센터에 들어오는 순간 모두 읽음 처리(요구사항에 부합)
  autoMarkAllReadOnEnter?: boolean;
}

export function NotificationsPage({
  onNavigate,
  notifications,
  onMarkRead,
  onMarkAllRead,
  autoMarkAllReadOnEnter = true,
}: NotificationsPageProps) {
  useEffect(() => {
    if (autoMarkAllReadOnEnter) onMarkAllRead();
  }, [autoMarkAllReadOnEnter, onMarkAllRead]);

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

  const unreadCount = notifications.filter((n) => !n.read).length;
  const urgentNotifications = notifications.filter((n) => n.urgent);
  const allNotifications = notifications;

  const unreadNotifications = notifications.filter((n) => !n.read);

  const renderNotificationCard = (notification: NotificationItem, forceRedStyle: boolean = false) => {
    const baseClass = notification.read
      ? "bg-gray-50"
      : forceRedStyle
        ? "border-red-200 bg-red-50"
        : "border-blue-200 bg-blue-50";

    return (
      <Card
        key={notification.id}
        className={`${baseClass} cursor-pointer`}
        onClick={() => {
          if (!notification.read) onMarkRead(notification.id);
        }}
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
                {notification.read && <Badge variant="outline">읽음</Badge>}
              </div>
              <h4 className="font-semibold mb-1">{notification.title}</h4>
              <p className="text-sm text-muted-foreground mb-2">{notification.message}</p>
              <p className="text-xs text-muted-foreground">{notification.time}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl mb-2">알림 센터</h2>
          <p className="text-muted-foreground">중요한 입찰 정보를 놓치지 마세요</p>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => onNavigate("profile")}>
            <Settings className="h-4 w-4 mr-2" />
            알림 설정
          </Button>

          <Button variant="default" onClick={onMarkAllRead} disabled={unreadCount === 0}>
            <CheckCheck className="h-4 w-4 mr-2" />
            모두 읽음
          </Button>
        </div>
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
          <TabsTrigger value="all">전체 ({allNotifications.length})</TabsTrigger>
          <TabsTrigger value="unread">읽지 않음 ({unreadCount})</TabsTrigger>
          <TabsTrigger value="urgent">긴급 ({urgentNotifications.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-3">
          {allNotifications.length === 0 ? (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">
                표시할 알림이 없습니다.
              </CardContent>
            </Card>
          ) : (
            allNotifications.map((n) => renderNotificationCard(n))
          )}
        </TabsContent>

        <TabsContent value="unread" className="space-y-3">
          {unreadNotifications.length === 0 ? (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">
                읽지 않은 알림이 없습니다.
              </CardContent>
            </Card>
          ) : (
            unreadNotifications.map((n) => renderNotificationCard(n))
          )}
        </TabsContent>

        <TabsContent value="urgent" className="space-y-3">
          {urgentNotifications.length === 0 ? (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">
                긴급 알림이 없습니다.
              </CardContent>
            </Card>
          ) : (
            urgentNotifications.map((n) => renderNotificationCard(n, true))
          )}
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
