package com.nara.aivleTK.domain.company;

import jakarta.persistence.*;
import lombok.*;

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
}
