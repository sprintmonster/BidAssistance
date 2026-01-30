import { api } from "./client";

export interface Bid {
    bidId: number;
    id: number;
    realId: string;
    name: string;
    startDate: string;
    endDate: string;
    openDate: string;
    region: string;
    organization: string;
    bidURL: string;
    bidReportURL: string;
    estimatePrice : bigint;

    minimumBidRate?: number | null;
    analysisResult?: string | null;
    bidDetail?: any | null;
    attachments?: any[];
}

function pickBidList(res: any): any[] {
    if (Array.isArray(res)) return res;
    if (Array.isArray(res?.data)) return res.data;
    if (Array.isArray(res?.data?.items)) return res.data.items;
    if (Array.isArray(res?.data?.content)) return res.data.content;
    return [];
}

export async function fetchBids(): Promise<Bid[]> {
    const res = await api<any>("/bids");
    return pickBidList(res) as Bid[];
}
