import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { CommunityBoard } from "./CommunityBoard";
import { PostDetail } from "./PostDetail";
import { NewPostForm } from "./NewPostForm";

import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs";

import { Info, Plus, Search as SearchIcon } from "lucide-react";
import type { NewPostDraftForm } from "./NewPostForm";
import type { Post, PostCategory, SortKey } from "../types/community";

import {
    createCommunityComment,
    createCommunityPost,
    deleteCommunityComment,
    deleteCommunityPost,
    fetchCommunityPost,
    fetchCommunityPosts,
    likeCommunityPost,
    unlikeCommunityPost,
    updateCommunityPost,
    uploadCommunityAttachments,
    fetchCommunityComments,
} from "../api/community";

type ViewMode = "list" | "detail" | "new";
type CategoryFilter = "all" | PostCategory;

const category_label: Record<CategoryFilter, string> = {
    all: "전체",
    question: "질문",
    info: "정보",
    review: "후기",
    discussion: "토론",
};

function is_authed_now() {
    return !!localStorage.getItem("userId");
}

function safe_user_id() {
    return localStorage.getItem("userId") || "";
}

function toValidId(v: unknown): number | null {
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? n : null;
}

/** ✅ 좋아요 localStorage (유저별) */
function likeStorageKey(userId?: string) {
    return `community_likes_v1:${userId ?? "guest"}`;
}

function loadLikedSet(userId?: string): Set<number> {
    try {
        const raw = localStorage.getItem(likeStorageKey(userId));
        if (!raw) return new Set();
        const arr = JSON.parse(raw);
        if (!Array.isArray(arr)) return new Set();
        return new Set(
            arr
                .map((x) => Number(x))
                .filter((n) => Number.isFinite(n) && n > 0)
        );
    } catch {
        return new Set();
    }
}

function saveLikedSet(userId: string | undefined, set: Set<number>) {
    try {
        localStorage.setItem(likeStorageKey(userId), JSON.stringify(Array.from(set)));
    } catch {
        // ignore
    }
}

function isLikedLocal(userId: string | undefined, postId: number) {
    return loadLikedSet(userId).has(postId);
}

/** likes 필드가 likes or likeCount 혼재일 수 있어서 안전 처리 */
function getLikes(p: any): number {
    const v = p?.likes ?? p?.likeCount ?? 0;
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
}

function setLikes(p: any, next: number) {
    // 둘 다 맞춰두면 UI/타입 혼재에도 안전
    return { ...p, likes: next, likeCount: next };
}

export function CommunityPage() {
    const navigate = useNavigate();

    const [view_mode, set_view_mode] = useState<ViewMode>("list");
    const [search_query, set_search_query] = useState("");
    const [category, set_category] = useState<CategoryFilter>("all");
    const [sort_key, set_sort_key] = useState<SortKey>("latest");

    const [posts, set_posts] = useState<Post[]>([]);
    const [counts, set_counts] = useState<Record<"all" | PostCategory, number>>({
        all: 0,
        question: 0,
        info: 0,
        review: 0,
        discussion: 0,
    });

    const [list_loading, set_list_loading] = useState(false);
    const [list_error, set_list_error] = useState<string | null>(null);

    const [selected_post, set_selected_post] = useState<Post | null>(null);
    const [detail_loading, set_detail_loading] = useState(false);
    const [detail_error, set_detail_error] = useState<string | null>(null);

    const [authed, set_authed] = useState(() => is_authed_now());
    const [current_user_id, set_current_user_id] = useState(() => safe_user_id());

    /** ✅ 게시글별 좋아요 연타 방지 */
    const [likeBusyByPost, setLikeBusyByPost] = useState<Record<number, boolean>>({});

    useEffect(() => {
        const sync = () => {
            set_authed(is_authed_now());
            set_current_user_id(safe_user_id());
        };
        sync();

        window.addEventListener("focus", sync);
        document.addEventListener("visibilitychange", sync);
        window.addEventListener("storage", sync);

        return () => {
            window.removeEventListener("focus", sync);
            document.removeEventListener("visibilitychange", sync);
            window.removeEventListener("storage", sync);
        };
    }, []);

    const go_login = () => navigate("/login", { state: { from: "/community" } });

    const load_list = async (opts?: { q?: string; c?: CategoryFilter; s?: SortKey }) => {
        set_list_loading(true);
        set_list_error(null);

        try {
            const c = opts?.c ?? category;
            const api_category = c === "all" ? undefined : c;
            const q = (opts?.q ?? search_query).trim() || undefined;
            const s = opts?.s ?? sort_key;

            const data = await fetchCommunityPosts({
                category: api_category,
                q,
                sort: s,
                page: 1,
                size: 50,
            });

            set_posts(data.items);

            if (data.counts) {
                set_counts({ ...data.counts, all: data.counts.all });
            } else {
                const base: Record<"all" | PostCategory, number> = {
                    all: data.items.length,
                    question: 0,
                    info: 0,
                    review: 0,
                    discussion: 0,
                };

                data.items.forEach((p: Post) => {
                    base[p.category] += 1;
                });

                set_counts(base);
            }
        } catch (e: any) {
            set_list_error(e?.message || "게시글 목록을 불러오지 못했습니다.");
        } finally {
            set_list_loading(false);
        }
    };

    const load_detail = async (post_id: number) => {
        set_detail_loading(true);
        set_detail_error(null);

        try {
            const [post, comments] = await Promise.all([
                fetchCommunityPost(post_id),
                fetchCommunityComments(post_id),
            ]);

            // ✅ 상세 진입 시에도 local 좋아요 상태를 덮어씌워서 버튼 상태가 안정적이게
            const pid = toValidId(post_id);
            const localLiked = pid ? isLikedLocal(current_user_id, pid) : false;

            const fixed: Post = {
                ...post,
                likedByMe: localLiked,
                comments,
                commentCount: comments.length,
                attachments: post.attachments ?? [],
            };

            set_selected_post(fixed);
        } catch (e: any) {
            set_detail_error(e?.message || "게시글을 불러오지 못했습니다.");
        } finally {
            set_detail_loading(false);
        }
    };

    useEffect(() => {
        const t = window.setTimeout(() => {
            void load_list();
        }, 250);

        return () => window.clearTimeout(t);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [category, sort_key, search_query]);

    const on_submit_search = (e: React.FormEvent) => {
        e.preventDefault();
        void load_list({ q: search_query });
    };

    const open_detail = (post: Post) => {
        const idNum = toValidId(post.id);
        if (idNum == null) {
            alert(`게시글 ID가 올바르지 않습니다. (id=${String(post.id)})`);
            return;
        }
        set_view_mode("detail");
        set_selected_post(null);
        void load_detail(idNum);
    };

    const back_to_list = () => {
        set_selected_post(null);
        set_view_mode("list");
    };

    const on_click_write = () => {
        if (!authed) return go_login();
        set_view_mode("new");
    };

    const can_edit_selected = useMemo(() => {
        if (!authed || !selected_post) return false;

        if ((selected_post as any).authorId) {
            return String((selected_post as any).authorId) === String(current_user_id);
        }

        const myName = localStorage.getItem("userName") || "";
        return !!myName && (selected_post as any).authorName === myName;
    }, [authed, selected_post, current_user_id]);

    const add_post = async (draft: NewPostDraftForm) => {
        if (!authed) return go_login();

        try {
            let attachment_ids: string[] | undefined;

            if (draft.files.length > 0) {
                const uploaded = await uploadCommunityAttachments(draft.files);
                attachment_ids = uploaded.map((a) => String(a.id));
            }

            const created = await createCommunityPost({
                title: draft.title,
                content: draft.content,
                category: draft.category,
                attachmentIds: attachment_ids,
            });

            const createdId = toValidId((created as any)?.id);
            if (createdId == null) {
                alert("작성된 게시글 ID가 올바르지 않습니다. 목록에서 다시 확인해주세요.");
                back_to_list();
                await load_list();
                return;
            }

            set_view_mode("detail");
            set_selected_post(null);

            await load_list();
            await load_detail(createdId);
        } catch (e: any) {
            alert(e?.message || "게시글 작성에 실패했습니다.");
        }
    };

    const add_comment = async (post_id: number, content: string) => {
        if (!authed) return go_login();

        const pid = Number(post_id);
        if (!Number.isFinite(pid) || pid <= 0) return;

        try {
            await createCommunityComment(pid, content);

            const comments = await fetchCommunityComments(pid);
            set_selected_post((prev) =>
                prev && Number((prev as any).id) === pid
                    ? { ...(prev as any), comments, commentCount: comments.length }
                    : prev
            );
        } catch (e: any) {
            alert(e?.message || "댓글 작성에 실패했습니다.");
        }
    };

    const update_post = async (
        post_id: number,
        patch: Partial<Pick<Post, "title" | "content" | "category">>
    ) => {
        if (!authed) return go_login();

        const pid = toValidId(post_id);
        if (pid == null) {
            alert(`게시글 ID가 올바르지 않습니다. (id=${String(post_id)})`);
            return;
        }

        try {
            const updated = await updateCommunityPost(pid, {
                title: patch.title,
                content: patch.content,
                category: patch.category,
            });

            // 좋아요 상태는 local 기준 유지
            const localLiked = isLikedLocal(current_user_id, pid);

            const fixed: Post = {
                ...(updated as any),
                likedByMe: localLiked,
                comments: (updated as any).comments ?? [],
                attachments: (updated as any).attachments ?? [],
            };

            set_selected_post((prev) => (prev && Number((prev as any).id) === pid ? fixed : prev));
            set_posts((prev) => prev.map((p) => (Number((p as any).id) === pid ? { ...(p as any), ...fixed } : p)));
        } catch (e: any) {
            alert(e?.message || "게시글 수정에 실패했습니다.");
        }
    };

    const delete_post = async (post_id: number) => {
        if (!authed) return go_login();

        const pid = toValidId(post_id);
        if (pid == null) {
            alert(`게시글 ID가 올바르지 않습니다. (id=${String(post_id)})`);
            return;
        }

        try {
            await deleteCommunityPost(pid);
            back_to_list();
            await load_list();
        } catch (e: any) {
            alert(e?.message || "게시글 삭제에 실패했습니다.");
        }
    };

    /** ✅ 좋아요 토글: 서버 응답 data를 안 읽고(localStorage + UI delta)로 처리 */
    const toggle_post_like = async (post_id: number) => {
        if (!authed) return go_login();

        const pid = toValidId(post_id);
        if (pid == null) {
            alert(`게시글 ID가 올바르지 않습니다. (id=${String(post_id)})`);
            return;
        }

        if (likeBusyByPost[pid]) return;

        const likedNow = isLikedLocal(current_user_id, pid);
        const delta = likedNow ? -1 : 1;

        try {
            setLikeBusyByPost((prev) => ({ ...prev, [pid]: true }));

            //  서버 호출 (응답 data는 무시)
            if (likedNow) await unlikeCommunityPost(pid);
            else await likeCommunityPost(pid);

            //  localStorage 업데이트
            const set = loadLikedSet(current_user_id);
            if (likedNow) set.delete(pid);
            else set.add(pid);
            saveLikedSet(current_user_id, set);

            //  리스트 UI 업데이트 (즉시)
            set_posts((prev) =>
                prev.map((p: any) => {
                    if (Number(p.id) !== pid) return p;
                    const nextLikes = Math.max(0, getLikes(p) + delta);
                    return { ...setLikes(p, nextLikes), likedByMe: !likedNow };
                })
            );

            //  상세 UI 업데이트 (즉시)
            set_selected_post((prev: any) => {
                if (!prev || Number(prev.id) !== pid) return prev;
                const nextLikes = Math.max(0, getLikes(prev) + delta);
                return { ...setLikes(prev, nextLikes), likedByMe: !likedNow };
            });

            //  여기서 load_detail(pid) 재호출하지 말 것 (느려짐 원인)
        } catch (e: any) {
            alert(e?.message || "좋아요 처리에 실패했습니다.");
        } finally {
            setLikeBusyByPost((prev) => ({ ...prev, [pid]: false }));
        }
    };

    const delete_comment = async (post_id: number, comment_id: string) => {
        if (!authed) return go_login();

        const pid = Number(post_id);
        const cid = Number(comment_id);
        if (!Number.isFinite(pid) || pid <= 0) return;
        if (!Number.isFinite(cid) || cid <= 0) return;

        try {
            await deleteCommunityComment(String(cid));

            const comments = await fetchCommunityComments(pid);
            set_selected_post((prev) =>
                prev && Number((prev as any).id) === pid
                    ? { ...(prev as any), comments, commentCount: comments.length }
                    : prev
            );
        } catch (e: any) {
            alert(e?.message || "댓글 삭제에 실패했습니다.");
        }
    };

    return (
        <div className="space-y-4">
            {view_mode === "list" && (
                <Card>
                    <CardHeader className="space-y-1">
                        <div className="flex items-start justify-between gap-4">
                            <div>
                                <CardTitle className="text-xl">커뮤니티</CardTitle>
                                <CardDescription>입찰 실무 노하우/질문/정보를 빠르게 공유하고 검색하세요.</CardDescription>
                            </div>

                            <div className="shrink-0">
                                <Button
                                    variant={authed ? "default" : "outline"}
                                    onClick={on_click_write}
                                    className="gap-2"
                                >
                                    {authed ? <Plus className="h-4 w-4" /> : null}
                                    {authed ? "글쓰기" : "로그인 후 글쓰기"}
                                </Button>
                            </div>
                        </div>
                    </CardHeader>

                    <CardContent className="space-y-3">
                        <Tabs value={category} onValueChange={(v) => set_category(v as CategoryFilter)}>
                            <div className="flex items-center justify-between gap-3">
                                <TabsList className="w-fit justify-start rounded-lg">
                                    <TabsTrigger value="all" className="flex-none px-3 rounded-md text-[13px]">
                                        전체 <span className="ml-1 text-xs opacity-70">{counts.all}</span>
                                    </TabsTrigger>
                                    <TabsTrigger value="question" className="flex-none px-3 rounded-md text-[13px]">
                                        질문 <span className="ml-1 text-xs opacity-70">{counts.question}</span>
                                    </TabsTrigger>
                                    <TabsTrigger value="info" className="flex-none px-3 rounded-md text-[13px]">
                                        정보 <span className="ml-1 text-xs opacity-70">{counts.info}</span>
                                    </TabsTrigger>
                                    <TabsTrigger value="review" className="flex-none px-3 rounded-md text-[13px]">
                                        후기 <span className="ml-1 text-xs opacity-70">{counts.review}</span>
                                    </TabsTrigger>
                                    <TabsTrigger value="discussion" className="flex-none px-3 rounded-md text-[13px]">
                                        토론 <span className="ml-1 text-xs opacity-70">{counts.discussion}</span>
                                    </TabsTrigger>
                                </TabsList>

                                <div className="hidden md:block text-xs text-gray-500 tabular-nums">
                                    {category_label[category]} · {posts.length}건
                                </div>
                            </div>
                        </Tabs>

                        <form onSubmit={on_submit_search} className="flex flex-col md:flex-row md:items-center gap-2">
                            <div className="relative flex-1">
                                <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                                <Input
                                    value={search_query}
                                    onChange={(e) => set_search_query(e.target.value)}
                                    placeholder="검색: 제목/내용/작성자"
                                    className="pl-9"
                                />
                            </div>

                            <div className="flex items-center gap-2">
                                <Button type="submit" className="whitespace-nowrap">
                                    검색하기
                                </Button>

                                <Select value={sort_key} onValueChange={(v) => set_sort_key(v as SortKey)}>
                                    <SelectTrigger className="w-[140px]">
                                        <SelectValue placeholder="정렬" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="latest">최신순</SelectItem>
                                        <SelectItem value="popular">인기순</SelectItem>
                                        <SelectItem value="views">조회순</SelectItem>
                                        <SelectItem value="comments">댓글순</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                        </form>

                        {!authed && (
                            <Alert className="bg-slate-50">
                                <Info />
                                <AlertTitle>게스트 모드</AlertTitle>
                                <AlertDescription>글쓰기/좋아요/댓글은 로그인 후 이용할 수 있습니다.</AlertDescription>
                            </Alert>
                        )}

                        {list_error && (
                            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                                {list_error}
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}

            {view_mode === "list" &&
                (list_loading ? (
                    <div className="text-sm text-gray-500 py-8 text-center">목록을 불러오는 중...</div>
                ) : (
                    <CommunityBoard posts={posts} onSelectPost={open_detail} />
                ))}

            {view_mode === "detail" &&
                (detail_loading ? (
                    <div className="text-sm text-gray-500 py-8 text-center">게시글을 불러오는 중...</div>
                ) : detail_error ? (
                    <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                        {detail_error}
                    </div>
                ) : selected_post ? (
                    <PostDetail
                        post={selected_post}
                        onBack={back_to_list}
                        onAddComment={add_comment}
                        onUpdatePost={update_post}
                        onDeletePost={delete_post}
                        onToggleLike={toggle_post_like}
                        onDeleteComment={delete_comment}
                        canEdit={can_edit_selected}
                        canInteract={authed}
                        onRequireAuth={go_login}
                        currentUserId={current_user_id}
                    />
                ) : null)}

            {view_mode === "new" && <NewPostForm onSubmit={add_post} onCancel={() => set_view_mode("list")} />}
        </div>
    );
}
