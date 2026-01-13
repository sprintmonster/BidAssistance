import { MessageSquare, Eye, ThumbsUp } from 'lucide-react';
import type { Post } from './CommunityPage';


interface CommunityBoardProps {
  posts: Post[];
  searchQuery: string;
  onSelectPost: (post: Post) => void;
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

export function CommunityBoard({ posts, searchQuery, onSelectPost }: CommunityBoardProps) {
  const filteredPosts = posts.filter(post =>
    post.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    post.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
    post.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="space-y-3">
      {filteredPosts.map((post) => (
        <div
          key={post.id}
          onClick={() => onSelectPost(post)}
          className="bg-white rounded-lg border border-gray-200 p-5 hover:border-blue-300 hover:shadow-md transition-all cursor-pointer"
        >
          <div className="flex items-start gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${categoryColors[post.category]}`}>
                  {categoryLabels[post.category]}
                </span>
                <span className="text-sm text-gray-500">{post.author}</span>
                <span className="text-sm text-gray-400">·</span>
                <span className="text-sm text-gray-500">{post.createdAt}</span>
              </div>

              <h3 className="text-lg font-semibold text-gray-900 mb-2">{post.title}</h3>
              <p className="text-gray-600 text-sm line-clamp-2 mb-3">{post.content}</p>

              <div className="flex items-center gap-4 text-sm text-gray-500">
                <div className="flex items-center gap-1">
                  <Eye className="w-4 h-4" />
                  <span>{post.views}</span>
                </div>
                <div className="flex items-center gap-1">
                  <MessageSquare className="w-4 h-4" />
                  <span>{post.comments.length}</span>
                </div>
                <div className="flex items-center gap-1">
                  <ThumbsUp className="w-4 h-4" />
                  <span>{post.likes}</span>
                </div>
              </div>

              {post.tags.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-3">
                  {post.tags.map((tag, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      ))}

      {filteredPosts.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">검색 결과가 없습니다.</p>
        </div>
      )}
    </div>
  );
}
