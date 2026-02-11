import { api } from "./client";

const MANUAL_LOGOUT_KEY = "manual_logout";

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

type CheckLoginData = {
	userId?: number | string;
	id?: number | string;
	name?: string;
	email?: string;
	role?: number;
	accessToken?: string;
	refreshToken?: string;
};

type CheckLoginApiResponse = {
	status: ApiStatus;
	message: string;
	data?: CheckLoginData;
};

function normalize_check_login_data(data: CheckLoginData): LoginSuccessData | null {
	const uid = data.userId ?? data.id;
	const name = data.name;

	if (uid == null || !name) return null;

	return {
		userId: uid,
		name,
		email: data.email,
		role: data.role,
		accessToken: data.accessToken,
		refreshToken: data.refreshToken,
	};
}

export function persistLogin(data: LoginSuccessData) {
	// 사용자가 '명시적으로 로그아웃'한 직후에는 자동 세션복구(checkLogin)로
	// 다시 로그인 상태로 돌아가는 것을 막기 위해 플래그를 제거한다.
	try {
		sessionStorage.removeItem(MANUAL_LOGOUT_KEY);
	} catch {
	}

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
	localStorage.removeItem("chatbot_messages"); // 챗봇 대화 내용도 삭제
}

export function login(email: string, password: string) {
	return api<LoginApiResponse>("/users/login", {
		method: "POST",
		body: JSON.stringify({ email, password }),
	});
}

export async function checkLogin(): Promise<LoginSuccessData | null> {
	// 사용자가 방금 '로그아웃'을 눌렀으면, 라우팅/리렌더 타이밍에 의해
	// checkLogin이 다시 세션을 복구(persistLogin)해버리는 것을 방지한다.
	try {
		if (sessionStorage.getItem(MANUAL_LOGOUT_KEY) === "1") return null;
	} catch {
	}

	try {
		const res = await api<CheckLoginApiResponse>(
			"/users/checkLogin",
			{ method: "GET" },
			{ on401: "ignore" },
		);

		if (res.status !== "success" || !res.data) return null;
		return normalize_check_login_data(res.data);
	} catch {
		return null;
	}
}

export async function logout() {
	try {
		await api<{ status: ApiStatus; message?: string }>("/users/logout", { method: "POST" });
	} catch {
	} finally {
		// 클라이언트에서 먼저 '명시적 로그아웃'을 표시해두면
		// 라우팅으로 checkLogin이 재실행되더라도 자동 로그인 복구를 막을 수 있다.
		try {
			sessionStorage.setItem(MANUAL_LOGOUT_KEY, "1");
		} catch {
		}

		clearLogin();
	}
}
