package com.nara.aivleTK.service;

import com.nara.aivleTK.dto.comment.CommentCreateRequest;
import com.nara.aivleTK.dto.comment.CommentResponse;

import java.util.List;

public interface CommentService {
    // null 넣으려면 Integer 써야함
    CommentResponse createComment(Integer bidId, Integer boardId, CommentCreateRequest request);

    void deleteComment(int commentId, int userId);

    List<CommentResponse> getCommentsByBid(int bidId);

    List<CommentResponse> getCommentsByBoard(int boardId);

    /**
     * 답변 채택 (질문 카테고리에서만 가능)
     * 
     * @param commentId 채택할 댓글 ID
     * @param userId    질문 작성자 ID (권한 확인용)
     * @return 채택된 댓글
     */
    CommentResponse adoptComment(int commentId, int userId);
}
