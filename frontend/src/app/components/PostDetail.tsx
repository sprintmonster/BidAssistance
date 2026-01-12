import { useState } from 'react';
import { ArrowLeft, Eye, ThumbsUp, MessageSquare, Send } from 'lucide-react';
import type { Post } from '../components/CommunityPage';

interface PostDetailProps {
  post: Post;
  onBack: () => void;
  onAddComment: (postId: string, content: string) => void;
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

export function PostDetail({ post, onBack, onAddComment }: PostDetailProps) {
  const [commentText, setCommentText] = useState('');

  const handleSubmitComment = () => {
    if (commentText.trim()) {
      onAddComment(post.id, commentText);
      setCommentText('');
    }
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
          <span className={`px-3 py-1 rounded text-sm font-medium ${categoryColors[post.category]}`}>
            {categoryLabels[post.category]}
          </span>
        </div>

        <h1 className="text-3xl font-bold text-gray-900 mb-4">{post.title}</h1>

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
          <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">{post.content}</p>
        </div>

        {post.tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-6">
            {post.tags.map((tag, idx) => (
              <span
                key={idx}
                className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-sm"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}

        <div className="flex items-center gap-4 pt-6 border-t border-gray-200">
          <button className="flex items-center gap-2 px-4 py-2 bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors">
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
