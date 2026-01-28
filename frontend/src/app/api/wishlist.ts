import { api } from "./client";
import type { WishlistItem } from "../types/wishlist";
import type { BidStage } from "../types/bid";
import { bid_stage_from_code, bid_stage_to_code } from "../types/bid";

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
	return bid_stage_from_code(v);
}

function parse_wishlist_item(userId: number, raw: AnyObj): WishlistItem {
	const hasWishlistShape =
		raw && (raw.bidId !== undefined || raw.userId !== undefined || raw.stage !== undefined);

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

async function patch_stage(userId: number, bidId: number, stage: BidStage): Promise<UpdateWishlistResponse> {
	const code = bid_stage_to_code(stage);
	return (await api(`/wishlist/stage/${userId}/${bidId}?stage=${code}`, {
		method: "PATCH",
	})) as UpdateWishlistResponse;
}

async function patch_memo_fallback(
	userId: number,
	bidId: number,
	wishlistId: number | undefined,
	memo: string,
): Promise<UpdateWishlistResponse> {
	const body: Record<string, any> = { memo };

	try {
		return (await api(`/wishlist?userId=${userId}&bidId=${bidId}`, {
			method: "PATCH",
			body: JSON.stringify(body),
		})) as UpdateWishlistResponse;
	} catch {
	}

	if (wishlistId && wishlistId > 0) {
		return (await api(`/wishlist/${wishlistId}`, {
			method: "PATCH",
			body: JSON.stringify(body),
		})) as UpdateWishlistResponse;
	}

	throw new Error("위시리스트 메모 업데이트 API 호출에 실패했습니다. (엔드포인트 확인 필요)");
}

export async function updateWishlist(params: {
	userId: number;
	bidId: number;
	wishlistId?: number;
	stage?: BidStage;
	memo?: string;
}): Promise<UpdateWishlistResponse> {
	const hasStage = params.stage !== undefined;
	const hasMemo = params.memo !== undefined;

	if (hasStage && !hasMemo) {
		return await patch_stage(params.userId, params.bidId, params.stage as BidStage);
	}

	if (!hasStage && hasMemo) {
		return await patch_memo_fallback(params.userId, params.bidId, params.wishlistId, String(params.memo));
	}

	if (hasStage && hasMemo) {
		await patch_stage(params.userId, params.bidId, params.stage as BidStage);
		return await patch_memo_fallback(params.userId, params.bidId, params.wishlistId, String(params.memo));
	}

	throw new Error("업데이트할 값이 없습니다.");
}
