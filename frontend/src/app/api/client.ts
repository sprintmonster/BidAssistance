const DOMAIN = import.meta.env.VITE_API_URL || "";
const BASE_URL = `${DOMAIN}/api`;

function isFormData(body: unknown): body is FormData {
	return typeof FormData !== "undefined" && body instanceof FormData;
}

export async function api<T>(url: string, options: RequestInit = {}): Promise<T> {
	const headers: Record<string, string> = {
		...((options.headers as Record<string, string>) ?? {}),
	};

	const token = localStorage.getItem("accessToken");
	if (token && !headers.Authorization) {
		headers.Authorization = `Bearer ${token}`;
	}

	if (options.body && isFormData(options.body)) {
		// multipart/form-data는 브라우저가 boundary 포함해 자동 세팅
	} else {
		if (!headers["Content-Type"]) headers["Content-Type"] = "application/json";
	}

	const res = await fetch(`${BASE_URL}${url}`, {
		...options,
		credentials: "include",
		headers,
	});

	if (res.status === 401) {
		localStorage.removeItem("userId");
		localStorage.removeItem("userName");
		localStorage.removeItem("name");
		localStorage.removeItem("email");
		localStorage.removeItem("role");
		localStorage.removeItem("accessToken");
		throw new Error("인증이 필요합니다.");
	}

	if (!res.ok) {
		const msg = await res.text().catch(() => "");
		throw new Error(msg || "API 오류");
	}

	return (await res.json()) as T;
}
