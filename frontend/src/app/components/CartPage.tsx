import { useEffect, useState } from "react";
import {
  fetchWishlist,
  updateWishlistStage,
  deleteWishlist,
} from "../api/wishlist";

export function CartPage({
  setGlobalLoading,
  showToast,
}: {
  setGlobalLoading: (v: boolean) => void;
  showToast: (msg: string, type: "success" | "error") => void;
}) {
  const [wishlist, setWishlist] = useState<any[]>([]);

  useEffect(() => {
    fetchWishlist().then(setWishlist);
  }, []);

  const changeStage = async (id: number, stage: string) => {
    try {
      setGlobalLoading(true);
      await updateWishlistStage(id, stage);
      setWishlist((prev) =>
        prev.map((w) => (w.wishlistId === id ? { ...w, stage } : w))
      );
      showToast("상태 변경 완료", "success");
    } catch {
      showToast("상태 변경 실패", "error");
    } finally {
      setGlobalLoading(false);
    }
  };

  const remove = async (id: number) => {
    try {
      setGlobalLoading(true);
      await deleteWishlist(id);
      setWishlist((prev) => prev.filter((w) => w.wishlistId !== id));
      showToast("삭제되었습니다", "success");
    } finally {
      setGlobalLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">장바구니</h2>

      {wishlist.map((w) => (
        <div key={w.wishlistId} className="border p-3 flex justify-between">
          <div>
            <div className="font-semibold">{w.title}</div>
            <div className="text-sm text-muted-foreground">
              {w.budget} · {w.deadline}
            </div>
          </div>

          <div className="flex gap-2">
            <select
              value={w.stage}
              onChange={(e) => changeStage(w.wishlistId, e.target.value)}
            >
              <option value="INTEREST">관심</option>
              <option value="REVIEW">검토중</option>
              <option value="DECIDED">참여결정</option>
              <option value="DOC_PREP">서류준비</option>
              <option value="SUBMITTED">제출완료</option>
              <option value="WON">낙찰</option>
              <option value="LOST">탈락</option>
            </select>
            <button onClick={() => remove(w.wishlistId)}>삭제</button>
          </div>
        </div>
      ))}
    </div>
  );
}
