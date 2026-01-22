import { api } from "./client";
import type { WishlistItem } from "../types/wishlist";
import type { BidStage } from "../types/bid";
import { isBidStage } from "../types/bid";

type ApiBase<T> = {
	status: "success" | "error";
	message?: string;
	data?: T;
};

type AnyObj = Record<string, any>;

function pick_list(res: any): any[] {
	const data = res?.data;
	if (Array.isArray(data)) return data;
	if (Array.isArray(data?.items)) return data.items;
	if (Array.isArray(data?.content)) return data.content;
	return [];
}

function pick_bid(source: AnyObj): AnyObj {
	// 서버가 Wishlist DTO에서 bid를 nested로 줄 수 있는 케이스 대응
	return (
		source?.bid ??
		source?.bidDto ??
		source?.bidInfo ??
		source?.bid_info ??
		source?.bid_detail ??
		source ??
		{}
	);
}

function to_string(v: any): string {
	if (v === null || v === undefined) return "";
	return String(v);
}

function parse_stage(v: any): BidStage {
	if (typeof v === "string" && isBidStage(v)) return v;
	return "INTEREST";
}

function parse_wishlist_item(userId: number, raw: AnyObj): WishlistItem {
	// 1) Wishlist 레코드 형태
	const hasWishlistShape = raw && (raw.bidId !== undefined || raw.userId !== undefined || raw.stage !== undefined);

	const bid = pick_bid(raw);

	if (hasWishlistShape) {
		return {
			id: Number(raw.id),
			userId: Number(raw.userId ?? userId),
			bidId: Number(raw.bidId ?? bid.id ?? raw.bid_id ?? raw.bidID),
			stage: parse_stage(raw.stage),

			decidedAt: raw.decidedAt ? to_string(raw.decidedAt) : undefined,
			submittedAt: raw.submittedAt ? to_string(raw.submittedAt) : undefined,
			resultAt: raw.resultAt ? to_string(raw.resultAt) : undefined,
			memo: raw.memo ? to_string(raw.memo) : undefined,

			realId: to_string(bid.realId ?? bid.bidRealId ?? bid.real_id ?? ""),
			title: to_string(bid.name ?? bid.title ?? ""),
			agency: to_string(bid.organization ?? bid.agency ?? ""),
			baseAmount: bid.estimatePrice ?? bid.baseAmount ?? "",
			bidStart: to_string(bid.startDate ?? bid.bidStart ?? ""),
			bidEnd: to_string(bid.endDate ?? bid.bidEnd ?? ""),
			openTime: to_string(bid.openDate ?? bid.openTime ?? ""),
			region: to_string(bid.region ?? ""),
		};
	}

	return {
		id: -1,
		userId,
		bidId: Number(raw.id),
		stage: parse_stage(raw.stage),

		realId: to_string(raw.realId ?? ""),
		title: to_string(raw.name ?? raw.title ?? ""),
		agency: to_string(raw.organization ?? raw.agency ?? ""),
		baseAmount: raw.estimatePrice ?? raw.baseAmount ?? "",
		bidStart: to_string(raw.startDate ?? raw.bidStart ?? ""),
		bidEnd: to_string(raw.endDate ?? raw.bidEnd ?? ""),
		openTime: to_string(raw.openDate ?? raw.openTime ?? ""),
		region: to_string(raw.region ?? ""),
	};
}

export async function fetchWishlist(userId: number): Promise<WishlistItem[]> {
	const res = (await api<ApiBase<any>>(`/wishlist/${userId}`, { method: "GET" })) as ApiBase<any>;
	const list = pick_list(res);
	return list.map((it) => parse_wishlist_item(userId, it));
}

type ToggleResponse = {
	status: "success" | "error";
	message?: string;
};

export async function toggleWishlist(userId: number, bidId: number): Promise<ToggleResponse> {
	return (await api(`/wishlist/toggle?userId=${userId}&bidId=${bidId}`, {
		method: "POST",
	})) as ToggleResponse;
}

type UpdateWishlistResponse = {
	status: "success" | "error";
	message?: string;
	data?: any;
};

export async function updateWishlist(
	params: {
		userId: number;
		bidId: number;
		wishlistId?: number;
		stage?: BidStage;
		memo?: string;
	},
): Promise<UpdateWishlistResponse> {
	const body: Record<string, any> = {};
	if (params.stage) body.stage = params.stage;
	if (params.memo !== undefined) body.memo = params.memo;

	try {
		return (await api(`/wishlist?userId=${params.userId}&bidId=${params.bidId}`, {
			method: "PATCH",
			body: JSON.stringify(body),
		})) as UpdateWishlistResponse;
	} catch {
		// fallthrough
	}

	if (params.wishlistId && params.wishlistId > 0) {
		return (await api(`/wishlist/${params.wishlistId}`, {
			method: "PATCH",
			body: JSON.stringify(body),
		})) as UpdateWishlistResponse;
	}

	// 둘 다 실패
	throw new Error("위시리스트 업데이트 API 호출에 실패했습니다. (엔드포인트 확인 필요)");
}
