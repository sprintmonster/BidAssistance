// types/wishlist.ts
import type { BidStage } from "./bid";

export type WishlistItem = {
    bidId: number;
    realId: string;
    title: string;
    agency: string;
    baseAmount: number | string; // BigInteger 대응(백엔드가 문자열로 줄 수도 있어서)
    bidStart: string;            // LocalDateTime -> ISO/string
    bidEnd: string;              // LocalDateTime -> ISO/string
    openTime: string;            // LocalDateTime -> ISO/string
    region: string;

    // 서버 스펙에 없음: 기존 UI용이라면 optional로
    stage?: BidStage;
};

