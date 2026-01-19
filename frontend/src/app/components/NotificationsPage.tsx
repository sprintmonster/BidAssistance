import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Bell, AlertCircle, FileText, RefreshCw, XCircle, Settings, CheckCheck } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";

export type NotificationItem = {
	id: number;
	type: string;
	title: string;
	message: string;
	time: string;
	read: boolean;
	urgent: boolean;
};

const STORAGE_KEY = "notifications.v1";

const DEFAULT_NOTIFICATIONS: NotificationItem[] = [
	{
		id: 1,
		type: "deadline",
		title: "마감 임박",
		message: "‘광주시 광산구 문화체육시설 신축’ 공고 마감이 24시간 남았습니다.",
		time: "방금",
		read: false,
		urgent: true,
	},
	{
		id: 2,
		type: "correction",
		title: "정정 공고",
		message: "‘대전시 유성구 복지센터 리모델링’ 정정사항이 등록되었습니다.",
		time: "1시간 전",
		read: false,
		urgent: false,
	},
	{
		id: 3,
		type: "reannouncement",
		title: "재공고",
		message: "관심 공고 중 1건이 재공고되었습니다.",
		time: "어제",
		read: true,
		urgent: false,
	},
];

export function NotificationsPage() {
	const [notifications, setNotifications] = useState<NotificationItem[]>(() => {
		try {
			const raw = localStorage.getItem(STORAGE_KEY);
			if (!raw) return DEFAULT_NOTIFICATIONS;
			const parsed = JSON.parse(raw) as NotificationItem[];
			return Array.isArray(parsed) ? parsed : DEFAULT_NOTIFICATIONS;
		} catch {
			return DEFAULT_NOTIFICATIONS;
		}
	});

	// 요구사항: 알림센터 진입 시 모두 읽음 처리(기본 true)
	useEffect(() => {
		setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	useEffect(() => {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(notifications));
	}, [notifications]);

	const unreadCount = useMemo(
		() => notifications.filter((n) => !n.read).length,
		[notifications],
	);

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

	const markRead = (id: number) => {
		setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, read: true } : n)));
	};

	const markAllRead = () => {
		setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
	};

	return (
		<div className="space-y-6">
			<Card>
				<CardHeader className="flex flex-row items-start justify-between gap-4">
					<div>
						<CardTitle className="flex items-center gap-2">
							<Bell className="h-5 w-5" />
							알림
							{unreadCount > 0 && <Badge variant="secondary">{unreadCount}</Badge>}
						</CardTitle>
						<CardDescription>마감, 정정, 재공고 등 중요한 이벤트를 확인하세요.</CardDescription>
					</div>

					<Button variant="outline" onClick={markAllRead}>
						<CheckCheck className="h-4 w-4" />
						모두 읽음
					</Button>
				</CardHeader>
				<CardContent>
					<Tabs defaultValue="all">
						<TabsList>
							<TabsTrigger value="all">전체</TabsTrigger>
							<TabsTrigger value="urgent">긴급</TabsTrigger>
							<TabsTrigger value="settings">설정</TabsTrigger>
						</TabsList>

						<TabsContent value="all" className="mt-4 space-y-3">
							{notifications.map((n) => (
								<button
									key={n.id}
									onClick={() => markRead(n.id)}
									className={[
										"w-full text-left rounded-xl border bg-white p-4 transition",
										"hover:bg-gray-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400",
										n.read ? "opacity-80" : "border-blue-200",
									].join(" ")}
								>
									<div className="flex items-start justify-between gap-4">
										<div className="flex items-start gap-3">
											<div className="mt-0.5">{getIcon(n.type)}</div>
											<div>
												<div className="flex items-center gap-2">
													<div className="font-semibold">{n.title}</div>
													{n.urgent && <Badge>긴급</Badge>}
													{!n.read && <Badge variant="secondary">NEW</Badge>}
												</div>
												<div className="text-sm text-gray-600 mt-1">{n.message}</div>
											</div>
										</div>
										<div className="text-xs text-gray-500 whitespace-nowrap">{n.time}</div>
									</div>
								</button>
							))}
						</TabsContent>

						<TabsContent value="urgent" className="mt-4 space-y-3">
							{notifications.filter((n) => n.urgent).map((n) => (
								<div key={n.id} className="rounded-xl border bg-white p-4">
									<div className="flex items-start justify-between gap-4">
										<div className="flex items-start gap-3">
											<div className="mt-0.5">{getIcon(n.type)}</div>
											<div>
												<div className="font-semibold">{n.title}</div>
												<div className="text-sm text-gray-600 mt-1">{n.message}</div>
											</div>
										</div>
										<div className="text-xs text-gray-500 whitespace-nowrap">{n.time}</div>
									</div>
								</div>
							))}
						</TabsContent>

						<TabsContent value="settings" className="mt-4">
							<div className="rounded-xl border bg-white p-4 space-y-4">
								<div className="flex items-center justify-between">
									<div className="flex items-center gap-2">
										<Settings className="h-4 w-4 text-gray-600" />
										<Label>마감 임박 알림</Label>
									</div>
									<Switch defaultChecked />
								</div>

								<div className="flex items-center justify-between">
									<div className="flex items-center gap-2">
										<Settings className="h-4 w-4 text-gray-600" />
										<Label>정정/재공고 알림</Label>
									</div>
									<Switch defaultChecked />
								</div>

								<div className="text-sm text-gray-500">
									* 실제 운영에서는 사용자 설정을 서버에 저장하도록 연결하세요.
								</div>
							</div>
						</TabsContent>
					</Tabs>
				</CardContent>
			</Card>
		</div>
	);
}
