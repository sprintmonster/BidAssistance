package com.nara.aivleTK.domain;

import jakarta.persistence.*;
import lombok.*;
import java.math.BigInteger;
import java.time.LocalDateTime;

@Entity
@Table(name = "bid")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class Bid {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column
    private int bidId;
    @Column
    private String bidRealId;
    @Column
    private String name;
    @Column
    private LocalDateTime startDate;
    @Column(nullable=true)
    private LocalDateTime endDate;
    @Column
    private LocalDateTime openDate;
    @Column
    private String region;
    @Column
    private BigInteger price;
    @Column
    private String organization;
    @Column
    private String bidFileName;
    @Column(name = "bid_URL")
    private String bidURL;
    @Column(name = "bid_report_URL")
    private String bidReportURL;
}
