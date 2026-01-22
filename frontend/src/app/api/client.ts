// src/app/api/client.ts
const DOMAIN = import.meta.env.VITE_API_URL || "";
const BASE_URL = `${DOMAIN}/api`;

export async function api<T>(url: string, options: RequestInit = {}): Promise<T> {
  const userId = localStorage.getItem("userId");

  const headers: Record<string, string> = {
    ...((options.headers as Record<string, string>) ?? {}),
  };

  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = headers["Content-Type"] ?? "application/json";
  }

  if (userId) headers["X-User-Id"] = userId;

  const res = await fetch(`${BASE_URL}${url}`, {
    ...options,
    credentials: "include",
    headers,
  });

  if (res.status === 401) {
    localStorage.removeItem("userId");
    window.location.href = "/";
    throw new Error("인증 만료");
  }

  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || "API 오류");
  }

  return res.json();
}
