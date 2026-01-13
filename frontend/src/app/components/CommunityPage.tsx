import { useMemo, useState } from "react";
import { CommunityBoard } from "./CommunityBoard";
import { PostDetail } from "./PostDetail";
import { NewPostForm } from "./NewPostForm";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

export type Comment = {
    id: string;
    author: string;
    content: string;
    createdAt: string;
    likes: number;
};
export type Attachment = {
    id: string;
    name: string;
    type: string;      // mime type e.g. image/png
    url: string;       // objectURL or uploaded URL
    size: number;
    isImage: boolean;
};

export type Post = {
    id: string;
    title: string;
    content: string;
    category: "question" | "info" | "review" | "discussion";
    // tags: string[];
    author: string;
    createdAt: string;
    views: number;
    likes: number;
    comments: Comment[];
    attachments: Attachment[];
};

type ViewMode = "list" | "detail" | "new";

export function CommunityPage() {
    const [viewMode, setViewMode] = useState<ViewMode>("list");
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedPost, setSelectedPost] = useState<Post | null>(null);

    const [posts, setPosts] = useState<Post[]>([
        {
            id: "p1",
            title: "입찰 서류 준비할 때 체크리스트 있을까요?",
            content: "처음 입찰 참여하는데 필수로 챙겨야 할 서류/실수 방지 팁 공유 부탁드립니다.",
            category: "question",
            // tags: ["입찰", "서류", "체크리스트"],
            author: "사용자A",
            createdAt: "2026-01-12",
            views: 12,
            likes: 3,
            comments: [],
            attachments: [],
        },
        {
            id: "p2",
            title: "마감 임박 공고 우선순위 정하는 기준 공유",
            content: "마감 임박 공고가 많을 때, 금액/지역/경쟁률 기반으로 우선순위 정하는 방법 공유합니다.",
            category: "info",
            // tags: ["마감임박", "우선순위"],
            author: "사용자B",
            createdAt: "2026-01-11",
            views: 30,
            likes: 8,
            comments: [],
            attachments: [],
        },
    ]);

    const filtered = useMemo(() => {
        const q = searchQuery.trim().toLowerCase();
        if (!q) return posts;
        return posts.filter(
            (p) =>
                p.title.toLowerCase().includes(q) ||
                p.content.toLowerCase().includes(q)
        );
    }, [posts, searchQuery]);

    const openDetail = (post: Post) => {
        // 조회수 증가한 "업데이트된 post"를 만든 뒤
        const nextPost = { ...post, views: post.views + 1 };

        // 목록 state 업데이트
        setPosts((prev) => prev.map((x) => (x.id === post.id ? nextPost : x)));

        // detail state도 업데이트된 post로 세팅
        setSelectedPost(nextPost);
        setViewMode("detail");
    };

    const backToList = () => {
        setSelectedPost(null);
        setViewMode("list");
    };

    const addComment = (postId: string, content: string) => {
        const now = new Date();
        const createdAt = now.toISOString().slice(0, 10);
        setPosts((prev) =>
            prev.map((p) => {
                if (p.id !== postId) return p;
                return {
                    ...p,
                    comments: [
                        ...p.comments,
                        {
                            id: `c_${Date.now()}`,
                            author: "사용자",
                            content,
                            createdAt,
                            likes: 0,
                        },
                    ],
                };
            })
        );

        // detail 화면에서 바로 갱신되게 selectedPost도 업데이트
        setSelectedPost((prev) => {
            if (!prev || prev.id !== postId) return prev;
            return {
                ...prev,
                comments: [
                    ...prev.comments,
                    {
                        id: `c_${Date.now()}`,
                        author: "사용자",
                        content,
                        createdAt,
                        likes: 0,
                    },
                ],
            };
        });
    };

    const addPost = (newPost: Omit<Post, "id" | "createdAt" | "views" | "likes" | "comments">) => {
        const createdAt = new Date().toISOString().slice(0, 10);
        const post: Post = {
            ...newPost,
            id: `p_${Date.now()}`,
            createdAt,
            views: 0,
            likes: 0,
            comments: [],
        };
        setPosts((prev) => [post, ...prev]);
        setViewMode("list");
    };

    return (
        <div className="space-y-4">
            {/* 상단 바 */}
            <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                    <CardTitle>커뮤니티</CardTitle>
                    {viewMode === "list" ? (
                        <Button onClick={() => setViewMode("new")}>글쓰기</Button>
                    ) : (
                        <Button variant="outline" onClick={() => setViewMode("list")}>
                            목록
                        </Button>
                    )}
                </CardHeader>

                {viewMode === "list" && (
                    <CardContent className="flex gap-2">
                        <Input
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="검색: 제목/내용/태그"
                        />
                    </CardContent>
                )}
            </Card>

            {/* 본문 */}
            {viewMode === "list" && (
                <CommunityBoard posts={filtered} searchQuery={searchQuery} onSelectPost={openDetail} />
            )}

            {viewMode === "detail" && selectedPost && (
                <PostDetail post={selectedPost} onBack={backToList} onAddComment={addComment} />
            )}

            {viewMode === "new" && (
                <NewPostForm onSubmit={addPost} onCancel={() => setViewMode("list")} />
            )}
        </div>
    );
}
