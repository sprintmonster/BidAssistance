import { useEffect, useState } from "react";

import { fetchWishlist, toggleWishlist } from "../api/wishlist";
import type { WishlistItem } from "../types/wishlist";

export function CartPage({
                             setGlobalLoading,
                             showToast,
                         }: {
    setGlobalLoading: (v: boolean) => void;
    showToast: (msg: string, type: "success" | "error") => void;
}) {
    const [wishlist, setWishlist] = useState<WishlistItem[]>([]);
    const userId = Number(localStorage.getItem("userId"));

    const loadWishlist = async () => {
        if (!Number.isFinite(userId)) {
            showToast("userId가 없습니다. 다시 로그인 해주세요.", "error");
            return;
        }
        const items = await fetchWishlist(userId);
        setWishlist(items);
    };

    useEffect(() => {
        void loadWishlist();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const onToggle = async (bidId: number) => {
        try {
            if (!Number.isFinite(userId)) {
                showToast("userId가 없습니다. 다시 로그인 해주세요.", "error");
                return;
            }
            setGlobalLoading(true);
            const res = await toggleWishlist(userId, bidId);
            // 토글 후에는 목록 다시 로드(가장 확실)
            await loadWishlist();
            showToast(res.message || "처리 완료", "success");
        } catch {
            showToast("찜 처리 실패", "error");
        } finally {
            setGlobalLoading(false);
        }
    };

    return (
        <div className="space-y-4">
            <h2 className="text-2xl font-bold">장바구니</h2>

            {wishlist.length === 0 ? (
                <div className="text-muted-foreground">찜한 공고가 없습니다.</div>
            ) : (
                wishlist.map((w) => (
                    <div key={w.bidId} className="border p-3 flex justify-between">
                        <div>
                            <div className="font-semibold">{w.title}</div>
                            <div className="text-sm text-muted-foreground">
                                {w.agency} · {w.baseAmount} · {w.bidEnd}
                            </div>
                        </div>

                        <div className="flex gap-2">
                            <button
                                className="px-3 py-1 border rounded"
                                onClick={() => void onToggle(w.bidId)}
                            >
                                찜 취소
                            </button>
                        </div>
                    </div>
                ))
            )}
        </div>
    );
}

// import { useEffect, useState } from "react";
//
// // import { deleteWishlist, fetchWishlist, updateWishlistStage } from "../api/wishlist";
// import { BID_STAGE_OPTIONS, isBidStage, type BidStage } from "../types/bid";
// import type { WishlistItem } from "../types/wishlist";
//
// export function CartPage({
// 	setGlobalLoading,
// 	showToast,
// }: {
// 	setGlobalLoading: (v: boolean) => void;
// 	showToast: (msg: string, type: "success" | "error") => void;
// }) {
// 	const [wishlist, setWishlist] = useState<WishlistItem[]>([]);
//
//     useEffect(() => {
//         const userId = Number(localStorage.getItem("userId")); // 네 프로젝트 기준으로 가져와야 함
//         if (!Number.isFinite(userId)) return;
//
//         fetchWishlist(userId).then(setWishlist);
//     }, []);
//
//     const changeStage = async (id: number, stage: BidStage) => {
// 		try {
// 			setGlobalLoading(true);
// 			await updateWishlistStage(id, stage);
// 			setWishlist((prev) =>
// 				prev.map((w) => (w.bidId === id ? { ...w, stage } : w))
// 			);
// 			showToast("상태 변경 완료", "success");
// 		} catch {
// 			showToast("상태 변경 실패", "error");
// 		} finally {
// 			setGlobalLoading(false);
// 		}
// 	};
//
// 	return (
// 		<div className="space-y-4">
// 			<h2 className="text-2xl font-bold">장바구니</h2>
//
// 			{wishlist.map((w) => (
// 				<div key={w.wishlistId} className="border p-3 flex justify-between">
// 					<div>
// 						<div className="font-semibold">{w.title}</div>
// 						<div className="text-sm text-muted-foreground">
// 							{w.budget} · {w.deadline}
// 						</div>
// 					</div>
//
// 					<div className="flex gap-2">
// 						<select
// 							value={w.stage}
// 							onChange={(e) => {
// 								const next = e.target.value;
// 								if (!isBidStage(next)) return;
// 								changeStage(w.wishlistId, next);
// 							}}
// 						>
// 							{BID_STAGE_OPTIONS.map((opt) => (
// 								<option key={opt.value} value={opt.value}>
// 									{opt.label}
// 								</option>
// 							))}
// 						</select>
// 					</div>
// 				</div>
// 			))}
// 		</div>
// 	);
// }
