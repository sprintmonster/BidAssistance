package com.nara.aivleTK.service;

import com.nara.aivleTK.dto.board.BoardRequest;
import com.nara.aivleTK.dto.board.BoardResponse;

public interface BoardService {

    BoardResponse creatPost(BoardRequest br);
    BoardResponse getPost(Integer id);
    BoardResponse updatePost(Integer id, BoardRequest br, Integer userId);
    void deletePost(Integer id, Integer userId);
}