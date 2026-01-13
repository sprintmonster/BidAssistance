package com.nara.aivleTK.dto;

import lombok.*;

import java.math.BigDecimal;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AnalysisResultDto {
    private Integer bidBidId;
    private BigDecimal goldenRate;
    private Long predictPrice;
    private BigDecimal avgRate;
    private String filepath;
    private String analysisContent;

}
