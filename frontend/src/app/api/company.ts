import { api } from "./client";

export type ApiStatus = "success" | "error";

export interface ApiResponse<T> {
	status: ApiStatus;
	message?: string;
	data?: T;
}

export type CompanyInfo = {
	companyId?: number;
	name: string;
	license?: string;
	performanceHistory?: string;
};

function to_str(v: unknown): string {
	if (typeof v === "string") return v;
	if (v == null) return "";
	return String(v);
}

function to_num(v: unknown): number | undefined {
	const n = Number(v);
	return Number.isFinite(n) ? n : undefined;
}

function normalize_company(res: any): CompanyInfo {
	const obj = res && typeof res === "object" ? (res.data ?? res) : {};
	return {
		companyId: to_num(obj.companyId ?? obj.id),
		name: to_str(obj.name),
		license: to_str(obj.license),
		performanceHistory: to_str(obj.performanceHistory),
	};
}

export async function getCompany(id: string | number): Promise<CompanyInfo> {
	const res = await api<any>(`/company/${id}`, { method: "GET" });
	return normalize_company(res);
}

export function upsertCompany(payload: {
	id?: number;
	name: string;
	license?: string;
	performanceHistory?: string;
}) {
	return api<ApiResponse<{ message?: string }>>("/company", {
		method: "POST",
		body: JSON.stringify(payload),
	});
}
