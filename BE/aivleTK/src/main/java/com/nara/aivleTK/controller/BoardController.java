package com.nara.aivleTK.controller;

import com.nara.aivleTK.domain.board.Board;
import com.nara.aivleTK.dto.ApiResponse;
import com.nara.aivleTK.dto.board.BoardListRequest;
import com.nara.aivleTK.dto.board.BoardListResponse;
import com.nara.aivleTK.dto.board.BoardRequest;
import com.nara.aivleTK.dto.board.BoardResponse;
import com.nara.aivleTK.dto.user.UserResponse;
import com.nara.aivleTK.exception.ResourceNotFoundException;
import com.nara.aivleTK.repository.BoardRepository;
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
    private final BoardRepository boardRepository;

    @PostMapping // 게시글 작성
    public ResponseEntity<ApiResponse<BoardResponse>> createPost(@RequestBody BoardRequest br, HttpSession session) {
        UserResponse user = (UserResponse) session.getAttribute("loginUser");
        br.setUserId(user.getId());

        return ResponseEntity.ok(ApiResponse.success("게시글이 작성되었습니다.", boardService.creatPost(br)));
    }

    @PostMapping("/{id}") // 게시글 업데이트
    public ResponseEntity<ApiResponse<BoardResponse>> updatePost(@PathVariable Integer id, @RequestBody BoardRequest br,
            HttpSession session) {
        UserResponse user = (UserResponse) session.getAttribute("loginUser");

        return ResponseEntity.ok(ApiResponse.success("게시글이 수정되었습니다.", boardService.updatePost(id, br, user.getId())));
    }

    @GetMapping("/{id}") // 게시글 상세보기
    public ResponseEntity<ApiResponse<BoardResponse>> getPost(@PathVariable Integer id) {
        return ResponseEntity.ok(ApiResponse.success(boardService.getPost(id)));
    }

    @DeleteMapping("/{id}") // 게시글 지우기
    public ResponseEntity<ApiResponse<Object>> deletePost(@PathVariable Integer id, HttpSession session) {
        UserResponse user = (UserResponse) session.getAttribute("loginUser");

        boardService.deletePost(id, user.getId());

        return ResponseEntity.ok(ApiResponse.success("게시글이 삭제되었습니다."));
    }

    @GetMapping("/posts")
    public ResponseEntity<ApiResponse<BoardListResponse>> boardList(@ModelAttribute BoardListRequest blr,
            HttpSession session) {
        UserResponse user = (UserResponse) session.getAttribute("loginUser");
        Integer userId = user != null ? user.getId() : null;

        return ResponseEntity.ok(ApiResponse.success("게시글 목록입니다.", boardService.getBoardList(blr, userId)));
    }

    @PostMapping("/posts/{id}/like")
    public ResponseEntity<ApiResponse<Object>> boardLike(@PathVariable Integer id) {
        Board board = boardRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("게시글을 찾을 수 없습니다."));
        boardRepository.addLikeCount(id);

        return ResponseEntity.ok(ApiResponse.success("좋아요를 눌렀습니다."));
    }

    @PostMapping("/posts/{id}/dislike")
    public ResponseEntity<ApiResponse<Object>> boardDislike(@PathVariable Integer id) {
        Board board = boardRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("게시글을 찾을 수 없습니다."));
        boardRepository.discardLikeCount(id);

        return ResponseEntity.ok(ApiResponse.success("좋아요를 취소했습니다."));
    }
}