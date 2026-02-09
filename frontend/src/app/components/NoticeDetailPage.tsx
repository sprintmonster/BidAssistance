import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { fetchCommunityPost, fetchCommunityComments, deleteCommunityPost, updateCommunityPost } from "../api/community";
import { PostDetail } from "./PostDetail";
import type { Post } from "../types/community";

export function NoticeDetailPage() {
    const navigate = useNavigate();
    const { id } = useParams();
    
    const [post, setPost] = useState<Post | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // 공지사항은 댓글/좋아요 인터랙션이 적으므로 기본적으로 로드
    // 다만 운영자가 아니면 수정/삭제 불가 (PostDetail 내부에서 처리됨)
    // 여기서는 데이터 로딩만 담당

    useEffect(() => {
        const postId = Number(id);
        if (!Number.isFinite(postId)) {
            setError("잘못된 공지사항 ID입니다.");
            setLoading(false);
            return;
        }

        (async () => {
            try {
                // 공지사항도 기술적으로는 Community Post임
                const [p, c] = await Promise.all([
                    fetchCommunityPost(postId),
                    fetchCommunityComments(postId)
                ]);

                setPost({
                    ...p,
                    comments: c,
                    commentCount: c.length,
                    attachments: p.attachments ?? []
                } as Post);
            } catch (e: any) {
                setError(e?.message || "공지사항을 불러오지 못했습니다.");
            } finally {
                setLoading(false);
            }
        })();
    }, [id]);

    const handleBack = () => {
        navigate("/notice");
    };

    // 운영자 기능 (삭제/수정)
    // 공지사항 삭제 시 notice 목록으로 이동
    const handleDelete = async (postId: number) => {
        try {
            await deleteCommunityPost(postId);
            navigate("/notice");
        } catch (e: any) {
            alert(e?.message || "삭제 실패");
        }
    };

    const handleUpdate = async (postId: number, patch: any) => {
        try {
            await updateCommunityPost(postId, patch);
            // 리로드
            const p = await fetchCommunityPost(postId);
            const c = await fetchCommunityComments(postId);
            setPost({ ...p, comments: c, commentCount: c.length, attachments: p.attachments ?? [] } as Post);
        } catch (e: any) {
            alert(e?.message || "수정 실패");
        }
    };

    if (loading) return <div className="py-20 text-center text-gray-500">불러오는 중...</div>;
    if (error) return <div className="py-20 text-center text-red-500">{error}</div>;
    if (!post) return <div className="py-20 text-center text-gray-500">공지사항을 찾을 수 없습니다.</div>;

    // 현재 로그인한 사용자 정보 (운영자 권한 체크용 등은 PostDetail 내부나 App에서 처리)
    const userId = localStorage.getItem("userId");
    
    // 운영자 여부 체크 (간단히 localStorage 기반, 실제 보안은 서버에서)
    // App.tsx의 isOperatorAccount 로직을 가져오거나, PostDetail에 isAdmin을 넘겨야 함.
    // 여기서는 App.tsx에서 props로 넘겨받는구조가 아니므로, 일단 localStorage 체크.
    const isAdmin = (() => {
        // App.tsx의 isOperatorAccount 로직 복사 또는 import 필요하지만
        // 간단히 email 체크 (실제 권한은 서버가 막음)
        const email = localStorage.getItem("email") || "";
        const role = localStorage.getItem("role") || "";
        return role.includes("ADMIN") || role.includes("OPERATOR"); 
    })();

    return (
        <PostDetail 
            post={post}
            onBack={handleBack}
            onAddComment={async () => { /* 공지사항 댓글 기능이 필요하다면 구현 */ }}
            onUpdatePost={handleUpdate}
            onDeletePost={handleDelete}
            onToggleLike={async () => { /* 공지사항 좋아요 */ }}
            onDeleteComment={async () => {}}
            canEdit={isAdmin} // 운영자만 수정 가능
            canInteract={!!userId}
            currentUserId={userId ?? undefined}
            isAdmin={isAdmin}
        />
    );
}
