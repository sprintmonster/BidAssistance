package com.nara.aivleTK.dto.board;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class BoardRequest {
    private String title;
    private String content;
    private Integer userId; // 유저
    private String category;
    private String filePath;
}
