// ../api/wishlist.ts
import { api } from "./client";
import type { WishlistItem } from "../types/wishlist";

type WishlistListResponse = {
    status: "success" | "error";
    message?: string;
    data?: {
        items?: Array<{
            bidId: number;
            realId: string;
            title: string;
            agency: string;
            baseAmount: unknown;
            bidStart: string;
            bidEnd: string;
            openTime: string;
            region: string;
        }>;
    };
};

export async function fetchWishlist(userId: number): Promise<WishlistItem[]> {
    const res = (await api(`/wishlist/${userId}`, { method: "GET" })) as WishlistListResponse;

    const items = res?.data?.items ?? [];

    return items.map((it) => ({
        bidId: it.bidId,
        realId: it.realId,
        title: it.title,
        agency: it.agency,
        baseAmount: String(it.baseAmount ?? ""),
        bidStart: it.bidStart,
        bidEnd: it.bidEnd,
        openTime: it.openTime,
        region: it.region,
        stage: "INTEREST",
    }));
}

type ToggleResponse = {
    status: "success" | "error";
    message?: string;
};

export async function toggleWishlist(userId: number, bidId: number): Promise<ToggleResponse> {
    return (await api("/wishlist/toggle", {
        method: "POST",
        body: JSON.stringify({ userId, bidId }),
    })) as ToggleResponse;
}

// await api("/wishlist/toggle", { method:"POST", body: { userId, bidId } })


// import { api } from "./client";
// import type { BidStage } from "../types/bid";
// import type { WishlistItem } from "../types/wishlist";
//
// export function fetchWishlist() {
// 	return api<WishlistItem[]>("/wishlist");
// }
//
// export function updateWishlistStage(wishlistId: number, stage: BidStage) {
// 	return api(`/wishlist/${wishlistId}/stage`, {
// 		method: "PATCH",
// 		body: JSON.stringify({ stage }),
// 	});
// }
//
// export function deleteWishlist(wishlistId: number) {
// 	return api(`/wishlist/${wishlistId}`, {
// 		method: "DELETE",
// 	});
// }
