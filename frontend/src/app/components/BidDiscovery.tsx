import { useEffect, useState } from "react";
import { fetchBids } from "../api/bids";
import { api } from "../api/client";

export function BidDiscovery({
  setGlobalLoading,
  showToast,
}: {
  setGlobalLoading: (v: boolean) => void;
  showToast: (msg: string, type: "success" | "error") => void;
}) {
  const [bids, setBids] = useState<any[]>([]);

  useEffect(() => {
    fetchBids().then(setBids);
  }, []);

  const addToCart = async (bidId: number) => {
    try {
      setGlobalLoading(true);
      await api("/wishlist", {
        method: "POST",
        body: JSON.stringify({ bidId }),
      });
      showToast("장바구니에 추가됨", "success");
    } catch {
      showToast("추가 실패", "error");
    } finally {
      setGlobalLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">공고 목록</h2>

      {bids.map((b) => (
        <div key={b.id} className="border p-4 flex justify-between">
          <div>
            <div className="font-semibold">{b.title}</div>
            <div className="text-sm text-muted-foreground">
              {b.agency} · {b.budget} · {b.deadline}
            </div>
          </div>
          <button onClick={() => addToCart(b.id)}>장바구니</button>
        </div>
      ))}
    </div>
  );
}
