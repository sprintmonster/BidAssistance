import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  ShoppingCart,
  Trash2,
  Calendar,
  DollarSign,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { useState } from "react";
import type { Page } from "../../types/navigation";

/* =====================
   Types
===================== */

export type BidStage =
  | "INTEREST"
  | "REVIEW"
  | "DECIDED"
  | "DOC_PREP"
  | "SUBMITTED"
  | "WON"
  | "LOST";

export type Bid = {
  id: number;
  title: string;
  agency: string;
  budget: string;
  budgetValue: number;
  deadline: string;
  stage: BidStage;
};

interface CartPageProps {
  cartItems: number[];
  bids: Bid[];
  onRemoveFromCart: (bidId: number) => void;
  onNavigate: (page: Page, bidId?: number) => void;
}

/* =====================
   Constants
===================== */

const STAGES: { key: BidStage; label: string }[] = [
  { key: "INTEREST", label: "관심" },
  { key: "REVIEW", label: "검토중" },
  { key: "DECIDED", label: "참여결정" },
  { key: "DOC_PREP", label: "서류준비" },
  { key: "SUBMITTED", label: "제출완료" },
  { key: "WON", label: "낙찰" },
  { key: "LOST", label: "탈락" },
];

/* =====================
   Component
===================== */

export function CartPage({
  cartItems,
  bids,
  onRemoveFromCart,
  onNavigate,
}: CartPageProps) {
  const [sortBy, setSortBy] = useState<"BUDGET" | "DEADLINE">("DEADLINE");

  const savedBids = bids.filter((b) => cartItems.includes(b.id));

  /* 상태 변경 (임시) */
  const updateStage = (bidId: number, stage: BidStage) => {
    const bid = bids.find((b) => b.id === bidId);
    if (bid) bid.stage = stage;
  };

  /* 정렬 적용 */
  const sortedBids = [...savedBids].sort((a, b) => {
    if (sortBy === "BUDGET") {
      return b.budgetValue - a.budgetValue;
    }
    return (
      new Date(a.deadline).getTime() -
      new Date(b.deadline).getTime()
    );
  });

  /* 상태별 개수 */
  const stageCount = STAGES.reduce((acc, s) => {
    acc[s.key] = savedBids.filter((b) => b.stage === s.key).length;
    return acc;
  }, {} as Record<BidStage, number>);

  if (savedBids.length === 0) {
    return (
      <Card>
        <CardContent className="py-16 flex flex-col items-center">
          <ShoppingCart className="h-14 w-14 text-muted-foreground mb-4" />
          <p className="mb-4">장바구니가 비어있습니다</p>
          <Button onClick={() => onNavigate("bids")}>
            공고 찾아보기
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* ===== Title ===== */}
      <div>
        <h2 className="text-3xl mb-1">장바구니</h2>
        <p className="text-muted-foreground">
          장바구니에 담은 공고를 관리하세요
        </p>
      </div>

      {/* ===== 진행 단계 요약 (위로 이동) ===== */}
      <Card>
        <CardContent className="py-4">
          <div className="grid grid-cols-2 md:grid-cols-7 gap-4 text-center">
            {STAGES.map((s) => (
              <div key={s.key}>
                <p className="text-sm text-muted-foreground">
                  {s.label}
                </p>
                <p className="text-xl font-bold">
                  {stageCount[s.key]}건
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* ===== 장바구니 공고 목록 ===== */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>장바구니 공고 목록</CardTitle>

          {/* 정렬 Select */}
          <Select
            value={sortBy}
            onValueChange={(v) =>
              setSortBy(v as "BUDGET" | "DEADLINE")
            }
          >
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="정렬" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="DEADLINE">
                마감일순
              </SelectItem>
              <SelectItem value="BUDGET">
                금액순
              </SelectItem>
            </SelectContent>
          </Select>
        </CardHeader>

        <CardContent className="space-y-3">
          {sortedBids.map((bid) => (
            <div
              key={bid.id}
              className="flex items-center justify-between border rounded-md p-3"
            >
              <div className="space-y-1">
                <p className="font-semibold">{bid.title}</p>
                <div className="text-sm text-muted-foreground flex gap-4">
                  <span className="flex items-center gap-1">
                    <DollarSign className="h-3 w-3" />
                    {bid.budget}
                  </span>
                  <span className="flex items-center gap-1">
                    <Calendar className="h-3 w-3" />
                    {bid.deadline}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                {/* 상태 변경 */}
                <Select
                  value={bid.stage}
                  onValueChange={(v) =>
                    updateStage(bid.id, v as BidStage)
                  }
                >
                  <SelectTrigger className="w-[120px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {STAGES.map((s) => (
                      <SelectItem key={s.key} value={s.key}>
                        {s.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => onRemoveFromCart(bid.id)}
                >
                  <Trash2 className="h-4 w-4 text-red-500" />
                </Button>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <Button variant="outline" onClick={() => onNavigate("bids")}>
        공고 더 찾아보기
      </Button>
    </div>
  );
}
