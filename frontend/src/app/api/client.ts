const DOMAIN = import.meta.env.VITE_API_URL || "";
const BASE_URL = `${DOMAIN}/api`;

function isFormData(body: unknown): body is FormData {
	return typeof FormData !== "undefined" && body instanceof FormData;
}

export async function api<T>(url: string, options: RequestInit = {}): Promise<T> {
	const headers: Record<string, string> = {
		...((options.headers as Record<string, string>) ?? {}),
	};

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
		// 세션 만료/미인증
		localStorage.removeItem("userId");
		localStorage.removeItem("userName");
		localStorage.removeItem("name");
		localStorage.removeItem("email");
		localStorage.removeItem("role");
		throw new Error("인증이 필요합니다.");
	}

	if (!res.ok) {
		const msg = await res.text().catch(() => "");
		throw new Error(msg || "API 오류");
	}

	return (await res.json()) as T;
}
