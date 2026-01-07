import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Search, Filter, MapPin, DollarSign, Calendar, Building, ShoppingCart, FileText } from "lucide-react";
import { Label } from "./ui/label";
import { Slider } from "./ui/slider";

interface BidDiscoveryProps {
  onNavigate: (page: string, bidId?: number) => void;
  onAddToCart: (bidId: number) => void;
}

export function BidDiscovery({ onNavigate, onAddToCart }: BidDiscoveryProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedRegion, setSelectedRegion] = useState("");
  const [selectedType, setSelectedType] = useState("");
  const [budgetRange, setBudgetRange] = useState([0, 100]);
  const [showFilters, setShowFilters] = useState(false);

  // Mock data
  const bids = [
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
      announcementDate: "2025-12-15",
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
      announcementDate: "2025-12-18",
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
      announcementDate: "2025-12-20",
    },
    {
      id: 4,
      title: "부산시 해운대구 주차장 건설",
      agency: "부산광역시 해운대구청",
      region: "부산",
      budget: "23억 원",
      budgetValue: 23,
      deadline: "2026-01-12",
      type: "공사",
      status: "진행중",
      announcementDate: "2025-12-22",
    },
    {
      id: 5,
      title: "대전시 유성구 복지센터 리모델링",
      agency: "대전광역시 유성구청",
      region: "대전",
      budget: "8억 원",
      budgetValue: 8,
      deadline: "2026-01-20",
      type: "공사",
      status: "진행중",
      announcementDate: "2025-12-25",
    },
    {
      id: 6,
      title: "광주시 광산구 문화체육시설 신축",
      agency: "광주광역시 광산구청",
      region: "광주",
      budget: "45억 원",
      budgetValue: 45,
      deadline: "2026-01-18",
      type: "공사",
      status: "진행중",
      announcementDate: "2025-12-28",
    },
  ];

  const filteredBids = bids.filter((bid) => {
    const matchesSearch = bid.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          bid.agency.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesRegion = !selectedRegion || bid.region === selectedRegion;
    const matchesType = !selectedType || bid.type === selectedType;
    const matchesBudget = bid.budgetValue >= budgetRange[0] && bid.budgetValue <= budgetRange[1];
    
    return matchesSearch && matchesRegion && matchesType && matchesBudget;
  });

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
        <h2 className="text-3xl mb-2">공고 찾기</h2>
        <p className="text-muted-foreground">지역과 금액 기준으로 입찰 공고를 검색하세요</p>
      </div>

      {/* Search and Filter Bar */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="공고명 또는 발주기관 검색..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Button
                variant={showFilters ? "default" : "outline"}
                onClick={() => setShowFilters(!showFilters)}
              >
                <Filter className="h-4 w-4 mr-2" />
                필터
              </Button>
            </div>

            {showFilters && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t">
                <div className="space-y-2">
                  <Label>지역</Label>
                  <Select value={selectedRegion} onValueChange={setSelectedRegion}>
                    <SelectTrigger>
                      <SelectValue placeholder="전체 지역" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">전체 지역</SelectItem>
                      <SelectItem value="서울">서울</SelectItem>
                      <SelectItem value="경기">경기</SelectItem>
                      <SelectItem value="인천">인천</SelectItem>
                      <SelectItem value="부산">부산</SelectItem>
                      <SelectItem value="대전">대전</SelectItem>
                      <SelectItem value="광주">광주</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>공고 유형</Label>
                  <Select value={selectedType} onValueChange={setSelectedType}>
                    <SelectTrigger>
                      <SelectValue placeholder="전체 유형" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">전체 유형</SelectItem>
                      <SelectItem value="공사">공사</SelectItem>
                      <SelectItem value="용역">용역</SelectItem>
                      <SelectItem value="물품">물품</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>예산 범위 (억원): {budgetRange[0]} ~ {budgetRange[1]}</Label>
                  <Slider
                    min={0}
                    max={100}
                    step={5}
                    value={budgetRange}
                    onValueChange={setBudgetRange}
                    className="mt-2"
                  />
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {filteredBids.length}개의 공고
          </p>
        </div>

        {filteredBids.map((bid) => {
          const daysLeft = getDaysUntilDeadline(bid.deadline);
          const isUrgent = daysLeft <= 3;

          return (
            <Card key={bid.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant={isUrgent ? "destructive" : "default"}>
                        {bid.type}
                      </Badge>
                      {isUrgent && (
                        <Badge variant="destructive">마감임박</Badge>
                      )}
                      <Badge variant="outline">{bid.status}</Badge>
                    </div>
                    <CardTitle className="text-xl mb-2">{bid.title}</CardTitle>
                    <CardDescription className="flex items-center gap-4 flex-wrap">
                      <span className="flex items-center gap-1">
                        <Building className="h-4 w-4" />
                        {bid.agency}
                      </span>
                      <span className="flex items-center gap-1">
                        <MapPin className="h-4 w-4" />
                        {bid.region}
                      </span>
                    </CardDescription>
                  </div>
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
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => onAddToCart(bid.id)}
                    >
                      <ShoppingCart className="h-4 w-4 mr-1" />
                      담기
                    </Button>
                    <Button
                      size="sm"
                      onClick={() => onNavigate("summary", bid.id)}
                    >
                      <FileText className="h-4 w-4 mr-1" />
                      상세보기
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
