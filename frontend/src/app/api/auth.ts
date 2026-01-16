import { api } from "./client";

export interface LoginResponse {
  accessToken: string;
  user: {
    id: number;
    email: string;
    name: string;
  };
}

export function login(email: string, password: string) {
  return api<LoginResponse>("/users/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function register(data: {
  email: string;
  password: string;
  name: string;
}) {
  return api("/users", {
    method: "POST",
    body: JSON.stringify(data),
  });
}
