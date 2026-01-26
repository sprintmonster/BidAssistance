import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
	Bell,
	AlertCircle,
	FileText,
	RefreshCw,
	XCircle,
	Settings,
	CheckCheck,
	Trash2,
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";

import { fetchWishlist } from "../api/wishlist";
import { deleteAlarm, fetchAlarms, type AlarmItem } from "../api/alarms";

export type NotificationItem = {
	id: number;
	bidId: number;
	type: string;
	title: string;
	message: string;
	time: string;
	read: boolean;
	urgent: boolean;
};

const READ_STORAGE_KEY = "notifications.read.v1";

function safe_json_parse<T>(raw: string | null, fallback: T): T {
	if (!raw) return fallback;
	try {
		return JSON.parse(raw) as T;
	} catch {
		return fallback;
	}
}

function get_user_id(): number | null {
	const raw = localStorage.getItem("userId");
	if (!raw) return null;
	const n = Number(raw);
	return Number.isFinite(n) ? n : null;
}

function now_ms(): number {
	return Date.now();
}

function to_relative_time(iso: string): string {
	const t = new Date(iso).getTime();
	if (!Number.isFinite(t)) return iso;

	const diff = Math.max(0, now_ms() - t);
	const sec = Math.floor(diff / 1000);
	if (sec < 30) return "방금";
	if (sec < 60) return `${sec}초 전`;

	const min = Math.floor(sec / 60);
	if (min < 60) return `${min}분 전`;

	const hour = Math.floor(min / 60);
	if (hour < 24) return `${hour}시간 전`;

	const day = Math.floor(hour / 24);
	if (day < 7) return `${day}일 전`;

	return new Date(iso).toLocaleDateString();
}

function infer_type_and_title(content: string): { type: string; title: string; urgent: boolean } {
	const c = content ?? "";
	// 백엔드가 content만 내려주는 형태를 가정하여 키워드 기반으로 분류
	if (/(마감|마감임박|마감 임박|D-?\d+)/i.test(c)) {
		return { type: "deadline", title: "마감 임박", urgent: true };
	}
	if (/(정정|변경|수정)/i.test(c)) {
		return { type: "correction", title: "정정 공고", urgent: false };
	}
	if (/(재공고|재입찰|재등록)/i.test(c)) {
		return { type: "reannouncement", title: "재공고", urgent: false };
	}
	if (/(낙찰|탈락|유찰|결과)/i.test(c)) {
		return { type: "result", title: "입찰 결과", urgent: true };
	}
	return { type: "general", title: "알림", urgent: false };
}

function load_read_map(): Record<string, boolean> {
	return safe_json_parse<Record<string, boolean>>(localStorage.getItem(READ_STORAGE_KEY), {});
}

function save_read_map(map: Record<string, boolean>) {
	localStorage.setItem(READ_STORAGE_KEY, JSON.stringify(map));
}

function to_notification_items(alarms: AlarmItem[], read_map: Record<string, boolean>): NotificationItem[] {
	return alarms
		.slice()
		.sort((a, b) => {
			const ta = new Date(a.date).getTime();
			const tb = new Date(b.date).getTime();
			return (Number.isFinite(tb) ? tb : 0) - (Number.isFinite(ta) ? ta : 0);
		})
		.map((a) => {
			const meta = infer_type_and_title(a.content);
			return {
				id: Number(a.alarmId),
				bidId: Number(a.bidId),
				type: meta.type,
				title: meta.title,
				message: a.content,
				time: to_relative_time(a.date),
				read: !!read_map[String(a.alarmId)],
				urgent: meta.urgent,
			};
		});
}

export function NotificationsPage() {
	const [items, setItems] = useState<NotificationItem[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const refresh = async () => {
		const userId = get_user_id();
		if (!userId) {
			setItems([]);
			setError("로그인이 필요합니다.");
			return;
		}

		setLoading(true);
		setError(null);

		try {
			// 1) 장바구니(위시리스트) 공고 id 목록
			const wishlist = await fetchWishlist(userId);
			const bid_ids = new Set(wishlist.map((w) => Number(w.bidId)).filter((n) => Number.isFinite(n)));

			// 2) 서버 알림 전체 조회 후, 장바구니 공고만 필터
			const alarms = await fetchAlarms(userId);
			const cart_scoped = alarms.filter((a) => bid_ids.has(Number(a.bidId)));

			// 3) 읽음 상태 merge
			const read_map = load_read_map();
			setItems(to_notification_items(cart_scoped, read_map));
		} catch (e) {
			const msg = e instanceof Error ? e.message : "알림을 불러오지 못했습니다.";
			setError(msg);
			setItems([]);
		} finally {
			setLoading(false);
		}
	};

	useEffect(() => {
		refresh();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	const unreadCount = useMemo(() => items.filter((n) => !n.read).length, [items]);

	const getIcon = (type: string) => {
		switch (type) {
			case "deadline":
				return <AlertCircle className="h-5 w-5 text-orange-600" />;
			case "correction":
				return <FileText className="h-5 w-5 text-blue-600" />;
			case "reannouncement":
				return <RefreshCw className="h-5 w-5 text-purple-600" />;
			case "result":
				return <XCircle className="h-5 w-5 text-red-600" />;
			default:
				return <Bell className="h-5 w-5 text-green-600" />;
		}
	};

	const markRead = (id: number) => {
		setItems((prev) => prev.map((n) => (n.id === id ? { ...n, read: true } : n)));
		const map = load_read_map();
		map[String(id)] = true;
		save_read_map(map);
	};

	const markAllRead = () => {
		setItems((prev) => prev.map((n) => ({ ...n, read: true })));
		const map = load_read_map();
		items.forEach((n) => {
			map[String(n.id)] = true;
		});
		save_read_map(map);
	};

	const removeOne = async (alarmId: number) => {
		try {
			await deleteAlarm(alarmId);
			setItems((prev) => prev.filter((n) => n.id !== alarmId));
			// 읽음 상태도 함께 정리
			const map = load_read_map();
			delete map[String(alarmId)];
			save_read_map(map);
		} catch (e) {
			const msg = e instanceof Error ? e.message : "알림 삭제에 실패했습니다.";
			setError(msg);
		}
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
						<CardDescription>
							장바구니(관심 공고)에 담긴 공고에 대해서만 알림이 표시됩니다.
						</CardDescription>
					</div>

					<div className="flex items-center gap-2">
						<Button variant="outline" onClick={refresh} disabled={loading}>
							<RefreshCw className={["h-4 w-4", loading ? "animate-spin" : ""].join(" ")} />
							새로고침
						</Button>
						<Button variant="outline" onClick={markAllRead} disabled={items.length === 0}>
							<CheckCheck className="h-4 w-4" />
							모두 읽음
						</Button>
					</div>
				</CardHeader>
				<CardContent>
					{error && (
						<div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
							{error}
						</div>
					)}

					<Tabs defaultValue="all">
						<TabsList>
							<TabsTrigger value="all">전체</TabsTrigger>
							<TabsTrigger value="urgent">긴급</TabsTrigger>
							<TabsTrigger value="settings">설정</TabsTrigger>
						</TabsList>

						<TabsContent value="all" className="mt-4 space-y-3">
							{loading && (
								<div className="text-sm text-gray-500">알림을 불러오는 중입니다...</div>
							)}
							{!loading && items.length === 0 && (
								<div className="text-sm text-gray-500">
									표시할 알림이 없습니다. (장바구니에 담긴 공고 기준)
								</div>
							)}
							{items.map((n) => (
								<div
									key={n.id}
									className={[
										"w-full rounded-xl border bg-white p-4 transition",
										n.read ? "opacity-80" : "border-blue-200",
									].join(" ")}
								>
									<div className="flex items-start justify-between gap-4">
										<button
											onClick={() => markRead(n.id)}
											className="flex-1 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 rounded-lg"
											type="button"
										>
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
										</button>

										<div className="flex items-center gap-2 shrink-0">
											<div className="text-xs text-gray-500 whitespace-nowrap">{n.time}</div>
											<Button
												size="icon"
												variant="ghost"
												aria-label="알림 삭제"
												onClick={() => removeOne(n.id)}
											>
												<Trash2 className="h-4 w-4" />
											</Button>
										</div>
									</div>
								</div>
							))}
						</TabsContent>

						<TabsContent value="urgent" className="mt-4 space-y-3">
							{items
								.filter((n) => n.urgent)
								.map((n) => (
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
							{!loading && items.filter((n) => n.urgent).length === 0 && (
								<div className="text-sm text-gray-500">긴급 알림이 없습니다.</div>
							)}
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
