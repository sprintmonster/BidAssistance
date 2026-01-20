package com.nara.aivleTK.controller;

import com.nara.aivleTK.domain.board.Board;
import com.nara.aivleTK.domain.user.User;
import com.nara.aivleTK.dto.ApiResponse;
import com.nara.aivleTK.repository.BoardRepository;
import com.nara.aivleTK.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/test")
@RequiredArgsConstructor
public class TestDataController {

    private final BoardRepository boardRepository;
    private final UserRepository userRepository;

    @PostMapping("/create-board-data")
    public ResponseEntity<ApiResponse<String>> createTestData() {
        User user = userRepository.findAll().stream().findFirst()
                .orElseThrow(() -> new RuntimeException("User not found. Please create a user first."));

        Board board1 = Board.builder()
                .title("첫 번째 질문입니다")
                .content("이것은 질문 카테고리의 테스트 게시글입니다. 내용이 100자를 넘어가면 미리보기가 잘리는지 확인하기 위한 긴 텍스트입니다. 추가로 더 많은 내용을 작성합니다.")
                .category("1")
                .user(user)
                .likeCount(5)
                .viewCount(20)
                .filePath("/uploads/test1.jpg")
                .build();

        Board board2 = Board.builder()
                .title("정보 공유 게시글")
                .content("유용한 정보를 공유합니다. 이 게시글은 info 카테고리입니다.")
                .category("2")
                .user(user)
                .likeCount(10)
                .viewCount(50)
                .filePath("/uploads/test2.jpg")
                .build();

        Board board3 = Board.builder()
                .title("리뷰 작성했습니다")
                .content("서비스 리뷰입니다. 리뷰 카테고리 테스트용 게시글입니다.")
                .category("3")
                .user(user)
                .likeCount(3)
                .viewCount(15)
                .filePath("")
                .build();

        Board board4 = Board.builder()
                .title("토론 주제입니다")
                .content("함께 토론하고 싶은 주제가 있어서 올립니다.")
                .category("4")
                .user(user)
                .likeCount(8)
                .viewCount(35)
                .filePath("/uploads/test4.jpg")
                .build();

        Board board5 = Board.builder()
                .title("인기 게시글")
                .content("조회수가 많은 인기 게시글입니다. 많은 사람들이 읽었습니다.")
                .category("1")
                .user(user)
                .likeCount(25)
                .viewCount(150)
                .filePath("/uploads/test5.jpg")
                .build();

        boardRepository.save(board1);
        boardRepository.save(board2);
        boardRepository.save(board3);
        boardRepository.save(board4);
        boardRepository.save(board5);

        return ResponseEntity.ok(ApiResponse.success("테스트 데이터 5개 생성 완료"));
    }
}
