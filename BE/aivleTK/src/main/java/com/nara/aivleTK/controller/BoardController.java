package com.nara.aivleTK.controller;

import com.nara.aivleTK.dto.board.BoardRequest;
import com.nara.aivleTK.dto.board.BoardResponse;
import com.nara.aivleTK.dto.user.UserResponse;
import com.nara.aivleTK.service.BoardService;
import jakarta.servlet.http.HttpSession;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/board")
@RequiredArgsConstructor
public class BoardController {
    private final BoardService boardService;

    @PostMapping // 게시글 작성
    public ResponseEntity<BoardResponse> createPost(@RequestBody BoardRequest br, HttpSession session) {
        UserResponse user = (UserResponse) session.getAttribute("loginUser");
        br.setUserId(user.getId());

        return ResponseEntity.ok(boardService.creatPost(br));
    }

    @PostMapping("/{id}") // 게시글 업데이트
    public ResponseEntity<BoardResponse> updatePost(@PathVariable Integer id, @RequestBody BoardRequest br, HttpSession session) {
        UserResponse user = (UserResponse) session.getAttribute("loginUser");

        return ResponseEntity.ok(boardService.updatePost(id, br, user.getId()));
    }

    @GetMapping("/{id}") // 게시글 상세보기
    public ResponseEntity<BoardResponse> getPost(@PathVariable Integer id) {
        return ResponseEntity.ok(boardService.getPost(id));
    }

    @DeleteMapping("/{id}") // 게시글 지우기
    public ResponseEntity<Void> deletePost(@PathVariable Integer id, HttpSession session) {
        UserResponse user = (UserResponse) session.getAttribute("loginUser");

        boardService.deletePost(id, user.getId());

        return ResponseEntity.noContent().build();
    }
}
