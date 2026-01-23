import type { BidStage } from "./bid";

export type WishlistItem = {
	id: number;
	userId: number;
	bidId: number;
	stage: BidStage;
	decidedAt?: string;   // Date -> ISO string (서버가 문자열로 내려준다고 가정)
	submittedAt?: string;
	resultAt?: string;
	memo?: string;

	realId: string;
	title: string;
	agency: string;
	baseAmount: number | string;
	bidStart: string;
	bidEnd: string;
	openTime: string;
	region: string;
};
