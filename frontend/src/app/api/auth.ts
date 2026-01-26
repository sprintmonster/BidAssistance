import { api } from "./client";

export type ApiStatus = "success" | "error";

export type LoginSuccessData = {
	userId: number | string;
	name: string;
	email?: string;

	// 백엔드가 내려줘도 세션 기반이면 프론트 인증에 사용하지 않음
	accessToken?: string;
	refreshToken?: string;

	role?: number;
};

export interface LoginApiResponse {
	status: ApiStatus;
	message: string;
	data?: LoginSuccessData;
}

export function persistLogin(data: LoginSuccessData) {
	localStorage.setItem("userId", String(data.userId));
	if (data.name != null) localStorage.setItem("userName", String(data.name));
	if (data.email != null) localStorage.setItem("email", String(data.email));
	if (typeof data.role === "number") localStorage.setItem("role", String(data.role));
	if (data.accessToken) localStorage.setItem("accessToken", String(data.accessToken));
	if (data.refreshToken) localStorage.setItem("refreshToken", String(data.refreshToken));
}

export function clearLogin() {
	localStorage.removeItem("userId");
	localStorage.removeItem("userName");
	localStorage.removeItem("name");
	localStorage.removeItem("email");
	localStorage.removeItem("role");
	localStorage.removeItem("accessToken");
	localStorage.removeItem("refreshToken");
}

export function login(email: string, password: string) {
	return api<LoginApiResponse>("/users/login", {
		method: "POST",
		body: JSON.stringify({ email, password }),
	});
}

export async function checkLogin(): Promise<LoginSuccessData | null> {
	const candidates = [
		"/users/checkLogin",
		"/users/login/check",
		"/users/check",
		"/users/me",
		"/users/session",
	];

	for (const path of candidates) {
		try {
			const res = await api<LoginApiResponse>(path, { method: "GET" });
			if (res.status === "success" && res.data) return res.data;
		} catch {
		}
	}
	return null;
}

export async function logout() {
	try {
		await api<{ status: ApiStatus; message?: string }>("/users/logout", { method: "POST" });
	} catch {
	} finally {
		clearLogin();
	}
}
