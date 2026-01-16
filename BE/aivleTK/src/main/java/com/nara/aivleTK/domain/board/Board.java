package com.nara.aivleTK.domain.board;

import com.nara.aivleTK.common.AutoTimeRecode;
import com.nara.aivleTK.domain.Comment;
import com.nara.aivleTK.domain.user.User;
import jakarta.persistence.*;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "board")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Board extends AutoTimeRecode {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "board_id")
    private Integer id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Builder.Default
    @OneToMany(mappedBy = "board", cascade = CascadeType.ALL)
    private List<Comment> comments = new ArrayList<>();

    @Column(nullable = false, length = 50)
    private String title;

    @Column(nullable = false, length = 50)
    private String content;

    @Column(columnDefinition = "BIT(3)")
    private Integer category;

    @Column(name = "like_count", nullable = false)
    private Integer likeCount;

    @Column(name = "view_count", nullable = false)
    private Integer viewCount;

    @Column(name = "file_path", nullable = true)
    private String filePath;
}
