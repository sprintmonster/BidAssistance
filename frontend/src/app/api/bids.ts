import { api } from "./client";

export interface Bid {
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
}

export function fetchBids() {
    return api<Bid[]>("/bids");
}
