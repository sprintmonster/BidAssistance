import { api } from "./client";

export type ApiStatus = "success" | "error";

export interface ApiResponse<T = unknown> {
	status: ApiStatus;
	message?: string;
	data?: T;
}

export interface AlarmItem {
	alarmId: number;
	userId: number;
	bidId: number;
	content: string;
	date: string; // LocalDateTime -> string(ISO)로 받는다고 가정
}

export interface AlarmListData {
	items: AlarmItem[];
}

function assertSuccess<T>(res: ApiResponse<T>, fallbackMsg: string): T {
	if (res.status !== "success") throw new Error(res.message || fallbackMsg);
	// data가 없을 수도 있는 응답(예: delete/email)도 있으니 캐스팅으로 처리
	return (res.data as T) ?? (undefined as unknown as T);
}

/**
 * 알림 목록 조회
 * GET /api/alarms/{userId}
 * response: { status:"success", data:{ items:[...] } }
 */
export async function fetchAlarms(userId: number | string): Promise<AlarmItem[]> {
	const res = await api<ApiResponse<AlarmListData>>(`/alarms/${userId}`);
	const data = assertSuccess(res, "알림 목록을 불러오지 못했습니다.");
	return data?.items ?? [];
}

/**
 * 알림 삭제
 * DELETE /api/alarms/{alarmId}
 * response: { status:"success", message:"알림이 삭제되었습니다." }
 */
export async function deleteAlarm(alarmId: number | string): Promise<{ message?: string }> {
	const res = await api<ApiResponse>(`/alarms/${alarmId}`, { method: "DELETE" });
	// success 시 message만 있는 형태라 data는 없을 수 있음
	if (res.status !== "success") throw new Error(res.message || "알림 삭제에 실패했습니다.");
	return { message: res.message };
}

/**
 * 이메일 알림 발송(테스트/관리자용)
 * POST /api/alarms/email
 * body: { email, subject, content }
 * response: { status:"success", message:"이메일 알림이 전송되었습니다." }
 */
export async function sendAlarmEmail(payload: {
	email: string;
	subject: string;
	content: string;
}): Promise<{ message?: string }> {
	const res = await api<ApiResponse>("/alarms/email", {
		method: "POST",
		body: JSON.stringify(payload),
	});
	if (res.status !== "success") throw new Error(res.message || "이메일 알림 발송에 실패했습니다.");
	return { message: res.message };
}

/**
 * POST /api/users/email_notify
 * body: { email, text }
 */
export async function sendUserEmailNotify(payload: {
	email: string;
	text: string;
}): Promise<{ message?: string }> {
	const res = await api<ApiResponse>("/users/email_notify", {
		method: "POST",
		body: JSON.stringify(payload),
	});
	if (res.status !== "success") throw new Error(res.message || "이메일 알림 전송에 실패했습니다.");
	return { message: res.message };
}

/**
 * POST /api/users/notification
 * body: { user_id, text }
 */
export async function sendUserNotification(payload: {
	user_id: number;
	text: string;
}): Promise<{ message?: string }> {
	const res = await api<ApiResponse>("/users/notification", {
		method: "POST",
		body: JSON.stringify(payload),
	});
	if (res.status !== "success") throw new Error(res.message || "알림 전송에 실패했습니다.");
	return { message: res.message };
}
