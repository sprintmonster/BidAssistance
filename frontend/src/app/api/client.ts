const DOMAIN = import.meta.env.VITE_API_URL || "";
const BASE_URL = `${DOMAIN}/api`;

function is_form_data(body: unknown): body is FormData {
	return typeof FormData !== "undefined" && body instanceof FormData;
}

function build_headers(options: RequestInit) {
	const headers: Record<string, string> = {
		...((options.headers as Record<string, string>) ?? {}),
	};

	const token = localStorage.getItem("accessToken");
	if (token && !headers.Authorization) headers.Authorization = `Bearer ${token}`;

	if (!(options.body && is_form_data(options.body))) {
		if (!headers["Content-Type"]) headers["Content-Type"] = "application/json";
	}

	return headers;
}

async function read_error_message(res: Response) {
	const txt = await res.text().catch(() => "");
	return txt || `API 오류 (HTTP ${res.status})`;
}

function clear_auth_storage() {
	localStorage.removeItem("userId");
	localStorage.removeItem("userName");
	localStorage.removeItem("name");
	localStorage.removeItem("email");
	localStorage.removeItem("role");
	localStorage.removeItem("accessToken");
}

export async function api<T>(url: string, options: RequestInit = {}): Promise<T> {
	const headers = build_headers(options);
	const method = String(options.method || "GET").toUpperCase();

	const res = await fetch(`${BASE_URL}${url}`, {
		...options,
		credentials: "include",
		headers,
	});

	if (res.status === 401) {
		/**
		 * 핵심 수정:
		 * - GET(특히 checkLogin 같은 "상태 확인")에서 401이 나왔다고 userId까지 지워버리면
		 *   테스트로그인/프론트 로그인 유지가 불가능해짐.
		 * - 쓰기(POST/PATCH/DELETE 등)에서 401이면 그때 정리하는 것이 안전.
		 */
		if (method !== "GET") clear_auth_storage();
		throw new Error("인증이 필요합니다.");
	}

	if (!res.ok) {
		throw new Error(await read_error_message(res));
	}

	return (await res.json()) as T;
}
