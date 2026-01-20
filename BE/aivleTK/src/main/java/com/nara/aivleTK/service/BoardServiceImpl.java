package com.nara.aivleTK.service;

import com.nara.aivleTK.domain.board.Board;
import com.nara.aivleTK.domain.user.User;
import com.nara.aivleTK.dto.board.BoardListRequest;
import com.nara.aivleTK.dto.board.BoardListResponse;
import com.nara.aivleTK.dto.board.BoardRequest;
import com.nara.aivleTK.dto.board.BoardResponse;
import com.nara.aivleTK.dto.board.BoardListItemResponse;
import com.nara.aivleTK.dto.board.CategoryCountsResponse;
import com.nara.aivleTK.repository.BoardRepository;
import com.nara.aivleTK.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
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

    public BoardListResponse getBoardList(BoardListRequest blr, Integer userId) {
        int page = ((blr.getPage() != null) && (blr.getPage() > 0)) ? blr.getPage() - 1 : 0;
        int size = (blr.getSize() != null) ? blr.getSize() : 10;

        Sort sort = Sort.by("createdAt").descending();
        if ("popular".equals(blr.getSort())) {
            sort = Sort.by("viewCount").descending();
        } else if ("likes".equals(blr.getSort())) {
            sort = Sort.by("likeCount").descending();
        }
        Pageable pageable = PageRequest.of(page, size, sort);

        Page<BoardResponse> boardPage = boardRepository.search(blr, pageable);

        List<BoardListItemResponse> items = boardPage.getContent().stream()
                .map(boardResponse -> {
                    Board board = boardRepository.findById(boardResponse.getId()).orElseThrow();
                    boolean likedByMe = false;
                    int commentCount = 0;
                    return BoardListItemResponse.from(board, likedByMe, commentCount);
                })
                .collect(Collectors.toList());

        CategoryCountsResponse counts = boardRepository.getCategoryCounts();

        return BoardListResponse.builder()
                .items(items)
                .page(page + 1)
                .size(size)
                .total(boardPage.getTotalElements())
                .counts(counts)
                .build();
    }
}
