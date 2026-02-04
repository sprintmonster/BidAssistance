import { useEffect, useState } from "react";
import { fetchCommunityPosts } from "../api/community";
import type { Post } from "../types/community";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Megaphone, Calendar, ChevronRight } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Skeleton } from "./ui/skeleton";

export function NoticePage() {
    const navigate = useNavigate();
    const [notices, setNotices] = useState<Post[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadNotices();
    }, []);

    const loadNotices = async () => {
        setLoading(true);
        try {
            // 'notice' 카테고리의 게시글만 불러옴
            const result = await fetchCommunityPosts({ category: "notice", size: 20 });
            setNotices(result.items);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-8 py-8">
            <div className="flex flex-col gap-2">
                <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
                    <Megaphone className="h-8 w-8 text-primary" />
                    공지사항
                </h1>
                <p className="text-muted-foreground">
                    서비스의 주요 변경사항과 안내를 확인하세요.
                </p>
            </div>

            <div className="space-y-4">
                {loading ? (
                    Array.from({ length: 3 }).map((_, i) => (
                        <Card key={i}>
                            <CardHeader>
                                <Skeleton className="h-6 w-3/4 mb-2" />
                                <Skeleton className="h-4 w-1/4" />
                            </CardHeader>
                        </Card>
                    ))
                ) : notices.length > 0 ? (
                    notices.map((notice) => (
                        <Card 
                            key={notice.id} 
                            className="hover:shadow-md transition cursor-pointer"
                            onClick={() => navigate(`/community`)} // 상세 페이지가 게시판 내부에 있으므로 일단 커뮤니티로 보내거나, 별도 라우트가 필요할 수 있음. 
                            // * 보통 공지사항 상세도 CommunityBoard 로직을 공유하므로 /community?tab=notice 로 보내거나 모달을 띄우는게 좋음.
                            // * 하지만 여기서는 일단 리스트를 보여주는게 목적.
                        >
                            <CardContent className="p-6 flex items-center justify-between">
                                <div className="space-y-1">
                                    <div className="flex items-center gap-2">
                                        <Badge variant="secondary" className="bg-primary/10 text-primary hover:bg-primary/20">공지</Badge>
                                        <h3 className="font-semibold text-lg">{notice.title}</h3>
                                    </div>
                                    <p className="text-muted-foreground text-sm line-clamp-1">
                                        {notice.contentPreview || "내용 미리보기"}
                                    </p>
                                    <div className="flex items-center gap-4 text-xs text-muted-foreground mt-2">
                                        <span className="flex items-center gap-1">
                                            <Calendar className="h-3 w-3" />
                                            {new Date(notice.createdAt).toLocaleDateString()}
                                        </span>
                                        <span>조회 {notice.views}</span>
                                    </div>
                                </div>
                                <ChevronRight className="h-5 w-5 text-muted-foreground" />
                            </CardContent>
                        </Card>
                    ))
                ) : (
                    <div className="text-center py-20 bg-slate-50 rounded-lg">
                        <Megaphone className="h-12 w-12 mx-auto text-muted-foreground mb-4 opacity-50" />
                        <h3 className="text-lg font-medium text-muted-foreground">등록된 공지사항이 없습니다.</h3>
                    </div>
                )}
            </div>
        </div>
    );
}
