import { api } from "./client";
import type { WishlistItem } from "../types/wishlist";

// 서버가 주는 위시리스트 아이템(실제 응답 기준)
type WishlistServerItem = {
    id: number;
    realId: string;
    name: string;
    organization: string;
    estimatePrice: unknown;
    startDate: string;
    endDate: string;
    openDate?: string;
    region?: string;
};

type WishlistListResponse = {
    status: "success" | "error";
    message?: string;
    data?: WishlistServerItem[] | { items?: WishlistServerItem[] };
};

export async function fetchWishlist(userId: number): Promise<WishlistItem[]> {
    const res = (await api(`/wishlist/${userId}`, { method: "GET" })) as WishlistListResponse;

    // ✅ data가 배열이거나, data.items 형태 둘 다 지원
    const list: WishlistServerItem[] = Array.isArray(res?.data)
        ? res.data
        : Array.isArray((res?.data as any)?.items)
            ? (res.data as any).items
            : [];

    return list.map((it) => ({
        bidId: it.id, // ✅ 서버는 id
        realId: String(it.realId ?? ""),
        title: String(it.name ?? ""),
        agency: String(it.organization ?? ""),
        baseAmount: String(it.estimatePrice ?? ""),
        bidStart: String(it.startDate ?? ""),
        bidEnd: String(it.endDate ?? ""),
        openTime: String(it.openDate ?? ""),
        region: String(it.region ?? ""),
        stage: "INTEREST",
    }));
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
