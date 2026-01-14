package com.nara.aivleTK.repository;

import com.nara.aivleTK.domain.board.Board;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface BoardRepository extends JpaRepository<Board, Integer> {

    @Modifying
    @Query("update Board b set b.viewCount = b.viewCount+1 where b.id =:id")
    void updateViewCount(@Param("id") Integer id);
}
