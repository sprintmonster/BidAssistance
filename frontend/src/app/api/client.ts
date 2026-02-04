const DOMAIN = import.meta.env.VITE_API_URL || "";
const BASE_URL = `${DOMAIN}/api`;

type Auth401Mode = "logout" | "ignore";

type ApiConfig = {
	on401?: Auth401Mode;
};

function isFormData(body: unknown): body is FormData {
	return typeof FormData !== "undefined" && body instanceof FormData;
}

function attach_auth_header(headers: Record<string, string>) {
	const token = localStorage.getItem("accessToken");
	if (!token) return;
	if (headers.Authorization) return;
	headers.Authorization = `Bearer ${token}`;
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

function looks_like_json(text: string) {
	const t = text.trim();
	if (!t) return false;
	return t.startsWith("{") || t.startsWith("[");
}

async function parse_success_response<T>(res: Response): Promise<T> {
	// 204 No Content 등: 바디가 없는 성공 응답
	if (res.status === 204) return undefined as T;

	const content_type = (res.headers.get("content-type") || "").toLowerCase();

	// 바디를 한 번만 읽고(중요) JSON/텍스트를 안전하게 처리
	const text = await res.text().catch(() => "");

	// 빈 바디 성공 응답
	if (!text.trim()) return undefined as T;

	// content-type이 JSON이거나, 실제로 JSON처럼 보이면 JSON 파싱 시도
	if (content_type.includes("application/json") || looks_like_json(text)) {
		try {
			return JSON.parse(text) as T;
		} catch {
			// 서버가 JSON이라 했는데 깨진 경우: 예외로 앱이 흔들리지 않도록 텍스트로 반환
			return text as unknown as T;
		}
	}

	// JSON이 아닌 성공 응답(텍스트 등)
	return text as unknown as T;
}

export async function api<T>(
	url: string,
	options: RequestInit = {},
	config: ApiConfig = {},
): Promise<T> {
	const headers: Record<string, string> = {
		...((options.headers as Record<string, string>) ?? {}),
	};

	attach_auth_header(headers);

	if (options.body && isFormData(options.body)) {
		// FormData일 때 Content-Type은 브라우저가 boundary 포함해서 설정해야 함
	} else {
		if (!headers["Content-Type"]) headers["Content-Type"] = "application/json";
	}

	const res = await fetch(`${BASE_URL}${url}`, {
		...options,
		credentials: "include",
		headers,
	});

	if (res.status === 401) {
		if ((config.on401 ?? "logout") === "logout") clear_auth_storage();
		throw new Error("인증이 필요합니다.");
	}

	if (!res.ok) {
		// 에러 메시지는 텍스트로 안전하게 파싱(서버가 JSON 에러를 주더라도 텍스트로 표시)
		const msg = await res.text().catch(() => "");
		throw new Error(msg || "API 오류");
	}

	return await parse_success_response<T>(res);
}
