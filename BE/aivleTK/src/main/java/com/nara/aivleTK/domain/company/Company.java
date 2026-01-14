package com.nara.aivleTK.domain.company;

import com.nara.aivleTK.domain.user.User;
import jakarta.persistence.*;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "company")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Company {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "company_id")
    private Integer id;

    @Column(length=15, nullable=false)
    private String name;

    @Column(length=200)
    private String license;

    @Column(name = "performance_history", length=200)
    private String performanceHistory;

    // 양방향 매핑 with User
    @OneToMany(mappedBy="company", cascade = CascadeType.ALL)
    private List<User> users = new ArrayList<>();
}
