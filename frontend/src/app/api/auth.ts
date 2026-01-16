// src/app/api/auth.ts

export type LoginSuccessResponse = {
	status: "success";
	message: string; // "로그인 성공"
	data: {
		userId: string; // "UUID"
		name: string; // "yujin"
		accessToken: string;
		refreshToken: string;
	};
};

export type LoginErrorResponse = {
	status: "error";
	message: string; // "이메일 또는 비밀번호가 올바르지 않습니다." 등
};

export type LoginResponse = LoginSuccessResponse | LoginErrorResponse;

function getApiBaseUrl() {
	// 예) VITE_API_BASE_URL=http://localhost:8080
	// 없으면 same-origin 기준으로 동작 (프록시 쓰는 경우)
	return (import.meta as any).env?.VITE_API_BASE_URL ?? "";
}

export async function login(email: string, password: string) {
	const baseUrl = getApiBaseUrl();

	const res = await fetch(`${baseUrl}/api/users/login`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({ email, password }),
	});

	let json: LoginResponse | null = null;
	try {
		json = (await res.json()) as LoginResponse;
	} catch {
		// JSON 파싱 실패(서버가 HTML/빈 body 반환 등)
	}

	// 서버 스펙이 status/message를 주므로 그걸 최우선으로 사용
	if (!res.ok) {
		const msg =
			(json && "message" in json && json.message) ||
			"서버 내부 오류가 발생했습니다. 관리자에게 문의하세요.";
		throw new Error(msg);
	}

	if (!json) {
		throw new Error("서버 응답을 처리할 수 없습니다.");
	}

	if (json.status === "error") {
		throw new Error(json.message || "로그인에 실패했습니다.");
	}

	return json.data; // { userId, name, accessToken, refreshToken }
}

