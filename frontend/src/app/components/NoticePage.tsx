import { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import type { Page } from "../../types/navigation";
import { Megaphone, Search } from "lucide-react";

type NoticeCategory = "전체" | "서비스" | "업데이트" | "점검" | "정책";

export interface NoticeItem {
    id: number;
    title: string;
    category: Exclude<NoticeCategory, "전체">;
    date: string; // YYYY-MM-DD
    pinned?: boolean;
    content: string;
}

const DEFAULT_NOTICES: NoticeItem[] = [
    {
        id: 1,
        title: "서비스 점검 안내 (01/10 02:00~04:00)",
        category: "점검",
        date: "2026-01-10",
        pinned: true,
        content:
            "안정적인 서비스 제공을 위해 시스템 점검이 진행됩니다.\n점검 시간 동안 일부 기능이 제한될 수 있습니다.\n\n- 점검 일시: 2026-01-10 02:00 ~ 04:00\n- 영향 범위: 로그인, 알림 일부 지연 가능\n\n이용에 불편을 드려 죄송합니다.",
    },
    {
        id: 2,
        title: "입찰 알림 필터 기능 업데이트",
        category: "업데이트",
        date: "2026-01-08",
        content:
            "알림 필터가 개선되었습니다.\n\n- 알림 유형별(마감/정정/재공고 등) 필터\n- 관심 지역 기반 우선순위\n- 읽지 않음만 보기\n\n설정은 [알림] 페이지에서 확인할 수 있습니다.",
    },
    {
        id: 3,
        title: "개인정보 처리방침 변경 안내",
        category: "정책",
        date: "2026-01-05",
        content:
            "개인정보 처리방침이 일부 개정되었습니다.\n\n- 수집 항목 문구 정비\n- 보관 및 파기 절차 보완\n\n자세한 내용은 [개인정보처리방침]을 참고해 주세요.",
    },
    {
        id: 4,
        title: "신규 기능: 공고 요약(Bid Summary) 베타 오픈",
        category: "서비스",
        date: "2025-12-28",
        content:
            "공고 상세 내용을 AI가 요약해주는 기능이 베타로 오픈되었습니다.\n\n- 주요 조건 요약\n- 리스크/체크포인트\n- 제출 서류 체크리스트\n\n피드백은 고객지원으로 보내주세요.",
    },
];

interface NoticePageProps {
    onNavigate: (page: Page, bidId?: number) => void;
    notices?: NoticeItem[];
}

export function NoticePage({ onNavigate, notices = DEFAULT_NOTICES }: NoticePageProps) {
    const [category, setCategory] = useState<NoticeCategory>("전체");
    const [query, setQuery] = useState("");
    const [openId, setOpenId] = useState<number | null>(null);

    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        const base = notices
            .slice()
            .sort((a, b) => {
                // pinned 먼저, 그 다음 날짜 내림차순
                const pa = a.pinned ? 1 : 0;
                const pb = b.pinned ? 1 : 0;
                if (pa !== pb) return pb - pa;
                return b.date.localeCompare(a.date);
            });

        return base.filter((n) => {
            const catOk = category === "전체" ? true : n.category === category;
            const qOk =
                !q ||
                n.title.toLowerCase().includes(q) ||
                n.content.toLowerCase().includes(q);
            return catOk && qOk;
        });
    }, [notices, category, query]);

    const categories: NoticeCategory[] = ["전체", "서비스", "업데이트", "점검", "정책"];

    return (
        <div className="space-y-6">
            <div className="flex items-start justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-bold flex items-center gap-2">
                        <Megaphone className="h-6 w-6" />
                        공지사항
                    </h2>
                    <p className="text-sm text-muted-foreground mt-1">
                        서비스 업데이트, 점검, 정책 변경 소식을 확인하세요.
                    </p>
                </div>

                {/*<div className="hidden sm:flex gap-2">*/}
                {/*    <Button variant="ghost" onClick={() => onNavigate("dashboard")}>*/}
                {/*        대시보드*/}
                {/*    </Button>*/}
                {/*    <Button variant="ghost" onClick={() => onNavigate("notifications")}>*/}
                {/*        알림*/}
                {/*    </Button>*/}
                {/*</div>*/}
            </div>

            {/* 검색/필터 */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">검색 및 필터</CardTitle>
                    <CardDescription>제목/내용으로 검색할 수 있습니다.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="space-y-2">
                        <Label htmlFor="notice-search">검색</Label>
                        <div className="relative">
                            <Search className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                            <Input
                                id="notice-search"
                                placeholder="예: 점검, 업데이트, 알림"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                className="pl-9"
                            />
                        </div>
                    </div>

                    <div className="flex flex-wrap gap-2">
                        {categories.map((c) => (
                            <Button
                                key={c}
                                size="sm"
                                variant={category === c ? "default" : "outline"}
                                onClick={() => setCategory(c)}
                            >
                                {c}
                            </Button>
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* 리스트 */}
            <div className="space-y-3">
                {filtered.length === 0 && (
                    <Card>
                        <CardContent className="py-8 text-center text-sm text-muted-foreground">
                            검색 결과가 없습니다.
                        </CardContent>
                    </Card>
                )}

                {filtered.map((n) => {
                    const isOpen = openId === n.id;
                    return (
                        <Card key={n.id} className={n.pinned ? "border-blue-200" : ""}>
                            <CardHeader className="pb-3">
                                <div className="flex items-start justify-between gap-3">
                                    <div className="space-y-1">
                                        <div className="flex items-center gap-2 flex-wrap">
                                            {n.pinned && <Badge>고정</Badge>}
                                            <Badge variant="secondary">{n.category}</Badge>
                                            <span className="text-xs text-muted-foreground">{n.date}</span>
                                        </div>

                                        <CardTitle className="text-lg">{n.title}</CardTitle>
                                    </div>

                                    <Button
                                        size="sm"
                                        variant="outline"
                                        onClick={() => setOpenId(isOpen ? null : n.id)}
                                    >
                                        {isOpen ? "닫기" : "자세히"}
                                    </Button>
                                </div>
                            </CardHeader>

                            {isOpen && (
                                <CardContent>
                                    <div className="whitespace-pre-line text-sm text-gray-700 leading-6">
                                        {n.content}
                                    </div>
                                </CardContent>
                            )}
                        </Card>
                    );
                })}
            </div>
        </div>
    );
}
