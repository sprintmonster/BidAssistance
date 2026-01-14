package com.nara.aivleTK.service;

import com.nara.aivleTK.domain.board.Board;
import com.nara.aivleTK.domain.user.User;
import com.nara.aivleTK.dto.board.BoardRequest;
import com.nara.aivleTK.dto.board.BoardResponse;
import com.nara.aivleTK.repository.BoardRepository;
import com.nara.aivleTK.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
@Transactional(readOnly=true)
public class BoardServiceImpl implements BoardService {

    private final BoardRepository boardRepository;
    private final UserRepository userRepository;

    @Transactional // 게시글 생성
    public BoardResponse creatPost(BoardRequest br) {
        User user = userRepository.findById(br.getUserId()).orElseThrow();

        Board board = Board.builder()
                .title(br.getTitle()).content(br.getContent())
                .user(user).category(br.getCategory())
                .filePath(br.getFilePath()).likeCount(0)
                .viewCount(0).build();

        return BoardResponse.from(boardRepository.save(board));
    }

    @Transactional // 게시글 불러오기
    public BoardResponse getPost(Integer id) {
        boardRepository.updateViewCount(id);
        Board board = boardRepository.findById(id).orElseThrow();

        return BoardResponse.from(board);
    }

    @Transactional // 게시글 업데이트
    public BoardResponse updatePost(Integer id, BoardRequest br, Integer userId) {
        Board board = boardRepository.findById(id).orElseThrow();
        User user = userRepository.findById(userId).orElseThrow();
        if ((!board.getUser().getId().equals(userId)) && (user.getRole() != 2)) {
            throw new IllegalStateException("수정 권한이 없습니다.");
        }

        board.setTitle(br.getTitle());
        board.setCategory(br.getCategory());
        board.setContent(br.getContent());
        board.setFilePath(br.getFilePath());

        return BoardResponse.from(boardRepository.save(board));
    }

    @Transactional // 게시글 삭제
    public void deletePost(Integer id, Integer userId) {
        Board board = boardRepository.findById(id).orElseThrow();
        User user = userRepository.findById(userId).orElseThrow();
        if ((board.getUser().getId().equals(userId)) || (user.getRole() != 2)) { // 관리자가 아니거나 작성자가 아니거나
            throw new IllegalStateException("삭제 권한이 없습니다.");
        }
        boardRepository.delete(board);
    }
}
