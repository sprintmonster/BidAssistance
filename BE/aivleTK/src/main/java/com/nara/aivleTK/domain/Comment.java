package com.nara.aivleTK.domain;

import com.nara.aivleTK.domain.board.Board;
import com.nara.aivleTK.domain.user.User;
import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name="comment")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder

public class Comment {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name="comment_id")
    private Integer commentId;

    @Column(name="comment_content", columnDefinition = "TEXT", nullable = false)
    private String commentContent;

    @Column(name="comment_date",nullable = false)
    private LocalDateTime commentCreateAt;

    @ManyToOne
    @JoinColumn(name="bid_id",nullable = true)
    private Bid bid;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "board_id", nullable = true)
    private Board board;

    @ManyToOne
    @JoinColumn(name="users_user_id",nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "parent_comment_id")
    private Comment parent;


    @PrePersist
    public void onCreate() {
        this.commentCreateAt = LocalDateTime.now();
    }
}

