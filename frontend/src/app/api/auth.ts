import { api } from "./client";

export type ApiStatus = "success" | "error";

export interface LoginSuccessData {
  userId: string;
  name: string;
  email: string;
  // accessToken: string;
  // refreshToken: string;
}

export interface LoginApiResponse {
  status: ApiStatus;
  message: string;
  data?: LoginSuccessData;
}

export function login(email: string, password: string) {
  return api<LoginApiResponse>("/users/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export interface RegisterData {
    email: string;
    name: string;
    password: string;
}

export function register(data: RegisterData) {
    return api<any>("/users", {
        method: "POST",
        body: JSON.stringify(data),
    });
}
