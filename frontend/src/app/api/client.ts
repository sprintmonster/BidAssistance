// const DOMAIN = import.meta.env.VITE_API_URL || "";
// const BASE_URL = `${DOMAIN}/api`;

const RAW = import.meta.env.VITE_API_URL || "";
const DOMAIN = RAW.replace(/\/+$/g, ""); // trailing slash 제거

// VITE_API_URL이 이미 /api를 포함하면 중복 방지
const BASE_URL = DOMAIN
    ? (DOMAIN.endsWith("/api") ? DOMAIN : `${DOMAIN}/api`)
    : "/api";

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
	localStorage.removeItem("refreshToken");
}

export async function api<T>(url: string, options: RequestInit = {}): Promise<T> {
	const headers = build_headers(options);

	const res = await fetch(`${BASE_URL}${url}`, {
		...options,
		credentials: "include",
		headers,
	});

	if (res.status === 401) {
		// clear_auth_storage();
		throw new Error("인증이 필요합니다.");
	}

	if (!res.ok) {
		throw new Error(await read_error_message(res));
	}

	return (await res.json()) as T;
}
