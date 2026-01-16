package com.nara.aivleTK.domain.user;

import com.nara.aivleTK.domain.company.Company;
import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDate;

@Entity
@Table(name = "user")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_id")
    private Integer id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "company_id")
    private Company company;

    @Column(length = 25, nullable = false)
    private String email;

    @Column(length = 10, nullable = false)
    private String name;

    @Column(length = 60, nullable = false)
    private String password;

    @Column(length = 50, nullable = false)
    private String question;

    @Column(length = 50, nullable = false)
    private String answer;

    private LocalDate birth;

    @Column(name = "role", columnDefinition = "BIT(2)")
    private Integer role; // 00: 일반 유저 01: 기업 10: 관리자 11: 휴면

    @Column(name = "tag", columnDefinition = "BIT(4)")
    private Integer tag; // 0000~0111: 일반 유저 태그, 1000~1111: 기업 유저 태그
}
