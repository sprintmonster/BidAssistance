import { useState, useEffect } from 'react';
import { ArrowLeft, Eye, ThumbsUp, MessageSquare, Send } from 'lucide-react';
import type { Post, Attachment } from "./CommunityPage";
import { Download, Image as ImageIcon, FileText as FileTextIcon } from "lucide-react";



interface PostDetailProps {
  post: Post;
  onBack: () => void;
  onAddComment: (postId: string, content: string) => void;
    onUpdatePost: (
        postId: string,
        patch: Partial<Pick<Post, "title" | "content" | "category" | "attachments">>
    ) => void;
    onDeletePost: (postId: string) => void;
    onToggleLike: (postId: string) => void;

}

const categoryLabels = {
  question: '질문',
  info: '정보',
  review: '후기',
  discussion: '토론',
};

const categoryColors = {
  question: 'bg-blue-100 text-blue-800',
  info: 'bg-green-100 text-green-800',
  review: 'bg-purple-100 text-purple-800',
  discussion: 'bg-orange-100 text-orange-800',
};

export function PostDetail({ post, onBack, onAddComment, onUpdatePost, onDeletePost, onToggleLike}: PostDetailProps) {
  const [commentText, setCommentText] = useState('');
    const [isEditing, setIsEditing] = useState(false);
    const [editTitle, setEditTitle] = useState(post.title);
    const [editContent, setEditContent] = useState(post.content);
    const [editCategory, setEditCategory] = useState<Post["category"]>(post.category);

    useEffect(() => {
        if (!isEditing) {
            setEditTitle(post.title);
            setEditContent(post.content);
            setEditCategory(post.category);
        }
    }, [post.id, isEditing, post.title, post.content, post.category]);

  const handleSubmitComment = () => {
    if (commentText.trim()) {
      onAddComment(post.id, commentText);
      setCommentText('');
    }
  };
    const startEdit = () => {
        setIsEditing(true);
        setEditTitle(post.title);
        setEditContent(post.content);
        setEditCategory(post.category);
    };

    const cancelEdit = () => {
        setIsEditing(false);
        // 원복
        setEditTitle(post.title);
        setEditContent(post.content);
        setEditCategory(post.category);
    };

    const saveEdit = () => {
        const t = editTitle.trim();
        const c = editContent.trim();
        if (!t || !c) return;

        onUpdatePost(post.id, {
            title: t,
            content: c,
            category: editCategory,
        });

        setIsEditing(false);
    };
    const handleDeletePost = () => {
        const ok = window.confirm("이 게시글을 삭제할까요? 삭제 후 복구할 수 없습니다.");
        if (!ok) return;

        // (선택) objectURL 정리 - 아래 3번 참고
        onDeletePost(post.id);
    };


    return (
    <div className="space-y-6">
      {/* Back Button */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
      >
        <ArrowLeft className="w-5 h-5" />
        <span>목록으로</span>
      </button>

      {/* Post Content */}
      <div className="bg-white rounded-lg border border-gray-200 p-8">
        <div className="mb-4">
          {/*<span className={`px-3 py-1 rounded text-sm font-medium ${categoryColors[post.category]}`}>*/}
          {/*  {categoryLabels[post.category]}*/}
          {/*</span>*/}

            {isEditing ? (
                <select
                    value={editCategory}
                    onChange={(e) => setEditCategory(e.target.value as Post["category"])}
                    className="px-3 py-2 border rounded-lg text-sm"
                >
                    <option value="question">질문</option>
                    <option value="info">정보</option>
                    <option value="review">후기</option>
                    <option value="discussion">토론</option>
                </select>
            ) : (
                <span className={`px-3 py-1 rounded text-sm font-medium ${categoryColors[post.category]}`}>
    {categoryLabels[post.category]}
  </span>
            )}

        </div>

          {/*{isEditing ? (*/}
          {/*    <input*/}
          {/*        value={editTitle}*/}
          {/*        onChange={(e) => setEditTitle(e.target.value)}*/}
          {/*        className="w-full text-2xl font-bold border rounded-lg px-3 py-2"*/}
          {/*    />*/}
          {/*) : (*/}
          {/*    <h1 className="text-3xl font-bold text-gray-900 mb-4">{post.title}</h1>*/}
          {/*)}*/}
          <div className="flex items-start justify-between gap-3 mb-4">
              <div className="min-w-0 flex-1">
                  {isEditing ? (
                      <input
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          className="w-full text-2xl font-bold border rounded-lg px-3 py-2"
                      />
                  ) : (
                      <h1 className="text-3xl font-bold text-gray-900 truncate">
                          {post.title}
                      </h1>
                  )}
              </div>

              <div className="shrink-0 flex gap-2">
                  {!isEditing && (
                      <button
                          type="button"
                          onClick={handleDeletePost}
                          className="px-3 py-2 rounded-lg border border-red-300 text-red-600 hover:bg-red-50"
                      >
                          삭제
                      </button>
                  )}

                  {isEditing ? (
                      <>
                          <button
                              type="button"
                              onClick={cancelEdit}
                              className="px-3 py-2 rounded-lg border"
                          >
                              취소
                          </button>
                          <button
                              type="button"
                              onClick={saveEdit}
                              className="px-3 py-2 rounded-lg bg-blue-600 text-white"
                          >
                              저장
                          </button>
                      </>
                  ) : (
                      <button
                          type="button"
                          onClick={startEdit}
                          className="px-3 py-2 rounded-lg border"
                      >
                          수정
                      </button>
                  )}
              </div>
          </div>

          <div className="flex items-center gap-4 text-sm text-gray-500 mb-6 pb-6 border-b border-gray-200">
          <span className="font-medium text-gray-700">{post.author}</span>
          <span>·</span>
          <span>{post.createdAt}</span>
          <span>·</span>
          <div className="flex items-center gap-1">
            <Eye className="w-4 h-4" />
            <span>{post.views}</span>
          </div>
        </div>

        <div className="prose max-w-none mb-6">
            {isEditing ? (
                <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="w-full border rounded-lg px-3 py-2 min-h-[240px]"
                />
            ) : (
                <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">{post.content}</p>
            )}
        </div>

        {/*{post.tags.length > 0 && (*/}
        {/*  <div className="flex flex-wrap gap-2 mb-6">*/}
        {/*    {post.tags.map((tag, idx) => (*/}
        {/*      <span*/}
        {/*        key={idx}*/}
        {/*        className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-sm"*/}
        {/*      >*/}
        {/*        #{tag}*/}
        {/*      </span>*/}
        {/*    ))}*/}
        {/*  </div>*/}
        {/*)}*/}
          {/* Attachments */}
          {post.attachments && post.attachments.length > 0 && (
              <div className="mb-6">
                  <div className="text-sm font-medium text-gray-700 mb-2">첨부파일</div>

                  {/* 이미지 썸네일 */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-3">
                      {post.attachments
                          .filter((a) => a.isImage)
                          .map((a) => (
                              <button
                                  key={a.id}
                                  type="button"
                                  onClick={() => window.open(a.url, "_blank")}
                                  className="group border rounded-lg overflow-hidden bg-gray-50 hover:shadow-sm transition text-left"
                                  title={a.name}
                              >
                                  <img src={a.url} alt={a.name} className="w-full h-32 object-cover" />
                                  <div className="p-2 text-xs text-gray-700 truncate">{a.name}</div>
                              </button>
                          ))}
                  </div>

                  {/* 파일 리스트 (이미지/비이미지 모두) */}
                  <div className="space-y-2">
                      {post.attachments.map((a) => (
                          <div
                              key={a.id}
                              className="flex items-center justify-between gap-3 border rounded-lg px-3 py-2"
                          >
                              <div className="flex items-center gap-2 min-w-0">
                                  {a.isImage ? (
                                      <ImageIcon className="w-4 h-4 text-gray-500" />
                                  ) : (
                                      <FileTextIcon className="w-4 h-4 text-gray-500" />
                                  )}
                                  <div className="min-w-0">
                                      <div className="text-sm text-gray-900 truncate">{a.name}</div>
                                      <div className="text-xs text-gray-500">
                                          {Math.round((a.size ?? 0) / 1024)} KB
                                      </div>
                                  </div>
                              </div>

                              <a
                                  href={a.url}
                                  download={a.name}
                                  className="shrink-0 inline-flex items-center gap-1 px-3 py-1.5 text-sm rounded-md border hover:bg-gray-50"
                                  onClick={(e) => e.stopPropagation()}
                              >
                                  <Download className="w-4 h-4" />
                                  다운로드
                              </a>
                          </div>
                      ))}
                  </div>
              </div>
          )}

        <div className="flex items-center gap-4 pt-6 border-t border-gray-200">
            <button
                type="button"
                onClick={() => onToggleLike(post.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                    post.likedByMe
                        ? "bg-blue-600 text-white hover:bg-blue-700"
                        : "bg-blue-50 text-blue-600 hover:bg-blue-100"
                }`}
            >
                <ThumbsUp className="w-5 h-5" />
                <span>좋아요 {post.likes}</span>
            </button>

        </div>
      </div>

      {/* Comments Section */}
      <div className="bg-white rounded-lg border border-gray-200 p-8">
        <div className="flex items-center gap-2 mb-6">
          <MessageSquare className="w-5 h-5 text-gray-700" />
          <h2 className="text-xl font-bold text-gray-900">
            댓글 {post.comments.length}
          </h2>
        </div>

        {/* Comment Input */}
        <div className="mb-8">
          <textarea
            value={commentText}
            onChange={(e) => setCommentText(e.target.value)}
            placeholder="댓글을 입력하세요..."
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            rows={4}
          />
          <div className="flex justify-end mt-2">
            <button
              onClick={handleSubmitComment}
              disabled={!commentText.trim()}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-4 h-4" />
              댓글 작성
            </button>
          </div>
        </div>

        {/* Comments List */}
        <div className="space-y-6">
          {post.comments.map((comment) => (
            <div key={comment.id} className="border-b border-gray-100 pb-6 last:border-b-0 last:pb-0">
              <div className="flex items-center gap-3 mb-2">
                <span className="font-medium text-gray-900">{comment.author}</span>
                <span className="text-sm text-gray-500">{comment.createdAt}</span>
              </div>
              <p className="text-gray-700 mb-3">{comment.content}</p>
              <button className="flex items-center gap-1 text-sm text-gray-500 hover:text-blue-600 transition-colors">
                <ThumbsUp className="w-4 h-4" />
                <span>{comment.likes}</span>
              </button>
            </div>
          ))}

          {post.comments.length === 0 && (
            <p className="text-center text-gray-500 py-8">
              첫 번째 댓글을 작성해보세요!
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
