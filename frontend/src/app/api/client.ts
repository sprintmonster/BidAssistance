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
		// do nothing
	} else {
		if (!headers["Content-Type"]) headers["Content-Type"] = "application/json";
	}

	const res = await fetch(`${BASE_URL}${url}`, {
		...options,
		credentials: "include",
		headers,
	});

	if (res.status === 401) {
		if ((config.on401 ?? "logout") === "logout") {
			localStorage.removeItem("userId");
			localStorage.removeItem("userName");
			localStorage.removeItem("name");
			localStorage.removeItem("email");
			localStorage.removeItem("role");
			localStorage.removeItem("accessToken");
			localStorage.removeItem("refreshToken");
		}
		throw new Error("인증이 필요합니다.");
	}

	if (!res.ok) {
		const msg = await res.text().catch(() => "");
		throw new Error(msg || "API 오류");
	}

	return (await res.json()) as T;
}
