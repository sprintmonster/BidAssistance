package com.nara.aivleTK.domain;

import jakarta.persistence.*;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@Table(name = "analysis_result")

public class AnalysisResult {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer analysisResultId;
    private Integer bidBidId;
    @Column(precision = 10, scale = 4)
    private BigDecimal goldenRate;
    private Long predictedPrice;
    @Column(precision = 10, scale = 4)
    private BigDecimal avgRate;
    private LocalDateTime analysisDate;

    @Column(length = 200)
    private String filepath;
    @Column(length = 1000)
    private String analysisContent;
}
