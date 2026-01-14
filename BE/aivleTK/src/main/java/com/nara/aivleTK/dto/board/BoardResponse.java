package com.nara.aivleTK.dto.board;

import com.nara.aivleTK.domain.board.Board;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@Builder
public class BoardResponse {
    private Integer id;
    private String title;
    private String userName;
    private String content;
    private Integer category;
    private Integer likeCount;
    private Integer viewCount;
    private String filePath;
    private LocalDateTime createdAt;

    public static BoardResponse from(Board board) {
        return BoardResponse.builder()
                .id(board.getId()).title(board.getTitle())
                .userName(board.getUser().getName()).content(board.getContent())
                .category(board.getCategory()).likeCount(board.getLikeCount())
                .viewCount(board.getViewCount()).filePath(board.getFilePath())
                .createdAt(board.getCreatedAt()).build();
    }
}
