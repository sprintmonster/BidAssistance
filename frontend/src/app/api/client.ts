const DOMAIN = import.meta.env.VITE_API_URL || "";
const BASE_URL = `${DOMAIN}/api`;

export async function api<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const token = localStorage.getItem("userId");

  const res = await fetch(`${BASE_URL}${url}`, {
    ...options,
      credentials: "include",
      headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  if (res.status === 401) {
    localStorage.removeItem("userId");
    window.location.href = "/";
    throw new Error("인증 만료");
  }

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || "API 오류");
  }

  return res.json();
}
