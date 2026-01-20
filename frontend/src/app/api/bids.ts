import { api } from "./client";

export interface Bid {
  id: string;
  title: string;
  agency: string;
  budget: string;
  deadline: string;
}

export function fetchBids() {
  return api<Bid[]>("/bids");
}
