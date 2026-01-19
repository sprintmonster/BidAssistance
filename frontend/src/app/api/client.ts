const BASE_URL = "/api";

export async function api<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const token = localStorage.getItem("accessToken");

  const res = await fetch(`${BASE_URL}${url}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  /* 🔥 토큰 만료 처리 */
  if (res.status === 401) {
    localStorage.removeItem("accessToken");
    window.location.href = "/";
    throw new Error("인증 만료");
  }

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || "API 오류");
  }

  return res.json();
}
