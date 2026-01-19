import { api } from "./client";
import type { BidStage } from "../components/CartPage";

export interface WishlistItem {
  wishlistId: number;
  bidId: number;
  title: string;
  agency: string;
  budget: string;
  budgetValue: number;
  deadline: string;
  stage: BidStage;
}

export function fetchWishlist() {
  return api<WishlistItem[]>("/wishlist");
}

export function updateWishlistStage(
  wishlistId: number,
  stage: BidStage
) {
  return api(`/wishlist/${wishlistId}/stage`, {
    method: "PATCH",
    body: JSON.stringify({ stage }),
  });
}

export function deleteWishlist(wishlistId: number) {
  return api(`/wishlist/${wishlistId}`, {
    method: "DELETE",
  });
}
