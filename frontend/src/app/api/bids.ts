import { api } from "./client";

export interface Bid {
  id: number;
  title: string;
  agency: string;
  budget: string;
  deadline: string;
}

export function fetchBids() {
  return api<Bid[]>("/bids");
}
