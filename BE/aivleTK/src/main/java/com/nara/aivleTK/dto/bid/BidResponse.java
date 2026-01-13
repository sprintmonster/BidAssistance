package com.nara.aivleTK.dto.bid;

import com.nara.aivleTK.domain.AnalysisResult;
import com.nara.aivleTK.domain.Bid;
import com.nara.aivleTK.dto.AnalysisResultDto;
import lombok.*;

import java.math.BigInteger;
import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
@Setter
@Builder
@AllArgsConstructor
public class BidResponse {
    private int id;
    private String realId;
    private String name;
    private LocalDateTime startDate;
    private LocalDateTime endDate;
    private LocalDateTime openDate;
    private String region;
    private BigInteger price;
    private String organization;
    private String bidURL;
    private String bidReportURL;
    private AnalysisResultDto analysisResult;

    public BidResponse(Bid bid){
        this.id = bid.getBidId();
        this.realId=bid.getBidRealId();
        this.name = bid.getName();
        this.startDate=bid.getStartDate();
        this.endDate=bid.getEndDate();
        this.openDate=bid.getOpenDate();
        this.region=bid.getRegion();
        this.price=bid.getPrice();
        this.organization=bid.getOrganization();
        this.bidURL=bid.getBidURL();
        this.bidReportURL=bid.getBidReportURL();
    }

}
