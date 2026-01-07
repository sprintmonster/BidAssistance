import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { ShoppingCart, Trash2, FileText, Calendar, DollarSign, Building } from "lucide-react";
import { Separator } from "./ui/separator";

interface CartPageProps {
  cartItems: number[];
  onRemoveFromCart: (bidId: number) => void;
  onNavigate: (page: string, bidId?: number) => void;
}

export function CartPage({ cartItems, onRemoveFromCart, onNavigate }: CartPageProps) {
  // Mock data - in real app, would fetch based on cartItems IDs
  const allBids = [
    {
      id: 1,
      title: "서울시 강남구 도로 보수공사",
      agency: "서울특별시 강남구청",
      region: "서울",
      budget: "35억 원",
      budgetValue: 35,
      deadline: "2026-01-08",
      type: "공사",
      status: "진행중",
    },
    {
      id: 2,
      title: "경기도 성남시 공공건물 신축공사",
      agency: "경기도 성남시청",
      region: "경기",
      budget: "87억 원",
      budgetValue: 87,
      deadline: "2026-01-15",
      type: "공사",
      status: "진행중",
    },
    {
      id: 3,
      title: "인천광역시 연수구 학교시설 개선공사",
      agency: "인천광역시 교육청",
      region: "인천",
      budget: "12억 원",
      budgetValue: 12,
      deadline: "2026-01-10",
      type: "공사",
      status: "진행중",
    },
  ];

  const savedBids = allBids.filter(bid => cartItems.includes(bid.id));
  const totalBudget = savedBids.reduce((sum, bid) => sum + bid.budgetValue, 0);

  const getDaysUntilDeadline = (deadline: string) => {
    const today = new Date("2026-01-06");
    const deadlineDate = new Date(deadline);
    const diffTime = deadlineDate.getTime() - today.getTime();
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays;
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl mb-2">관심 공고 장바구니</h2>
        <p className="text-muted-foreground">저장한 입찰 공고를 관리하세요</p>
      </div>

      {savedBids.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <ShoppingCart className="h-16 w-16 text-muted-foreground mb-4" />
            <h3 className="text-xl font-semibold mb-2">장바구니가 비어있습니다</h3>
            <p className="text-muted-foreground mb-6">관심있는 공고를 담아보세요</p>
            <Button onClick={() => onNavigate("bids")}>공고 찾아보���</Button>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Summary Card */}
          <Card className="bg-gradient-to-r from-blue-50 to-indigo-50">
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <p className="text-sm text-muted-foreground mb-1">저장된 공고</p>
                  <p className="text-3xl font-bold">{savedBids.length}건</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground mb-1">총 예산 규모</p>
                  <p className="text-3xl font-bold">{totalBudget}억 원</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground mb-1">긴급 마감</p>
                  <p className="text-3xl font-bold text-red-600">
                    {savedBids.filter(bid => getDaysUntilDeadline(bid.deadline) <= 3).length}건
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Saved Bids List */}
          <div className="space-y-4">
            {savedBids.map((bid) => {
              const daysLeft = getDaysUntilDeadline(bid.deadline);
              const isUrgent = daysLeft <= 3;

              return (
                <Card key={bid.id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge>{bid.type}</Badge>
                          {isUrgent && <Badge variant="destructive">마감임박</Badge>}
                          <Badge variant="outline">{bid.status}</Badge>
                        </div>
                        <CardTitle className="text-xl mb-2">{bid.title}</CardTitle>
                        <CardDescription>
                          <div className="flex items-center gap-4 flex-wrap">
                            <span className="flex items-center gap-1">
                              <Building className="h-4 w-4" />
                              {bid.agency}
                            </span>
                            <span>{bid.region}</span>
                          </div>
                        </CardDescription>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onRemoveFromCart(bid.id)}
                      >
                        <Trash2 className="h-4 w-4 text-red-600" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-between">
                      <div className="flex gap-6 text-sm">
                        <div className="flex items-center gap-2">
                          <DollarSign className="h-4 w-4 text-muted-foreground" />
                          <span className="font-semibold">{bid.budget}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Calendar className="h-4 w-4 text-muted-foreground" />
                          <span>마감: {bid.deadline}</span>
                          <span className={isUrgent ? "text-red-600 font-semibold" : "text-muted-foreground"}>
                            (D-{daysLeft})
                          </span>
                        </div>
                      </div>
                      <Button size="sm" onClick={() => onNavigate("summary", bid.id)}>
                        <FileText className="h-4 w-4 mr-1" />
                        상세보기
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          <Separator />

          <div className="flex justify-between items-center">
            <Button variant="outline" onClick={() => onNavigate("bids")}>
              공고 더 찾아보기
            </Button>
            <div className="text-sm text-muted-foreground">
              총 {savedBids.length}개 공고 · {totalBudget}억 원
            </div>
          </div>
        </>
      )}
    </div>
  );
}
