const BASE_URL = import.meta.env.VITE_API_URL ?? "/api";

export async function api<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const token = localStorage.getItem("accessToken");

  const res = await fetch(`${BASE_URL}${url}`, {
    ...options,
      credentials: "include",
      headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  /* 🔥 토큰 만료 처리 */
	if (res.status === 401 && token) {
    localStorage.removeItem("accessToken");
    window.location.href = "/";
    throw new Error("인증 만료");
  }

  if (!res.ok) {
		let msg = "API 오류";
		try {
			const json = await res.json();
			msg = (json as any)?.message || msg;
		} catch {
			try {
				const text = await res.text();
				msg = text || msg;
			} catch {
				// ignore
			}
		}
		throw new Error(msg);
  }

  return res.json();
}
