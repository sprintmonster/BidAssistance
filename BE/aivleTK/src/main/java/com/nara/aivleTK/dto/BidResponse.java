package com.nara.aivleTK.dto;

import com.nara.aivleTK.domain.Bid;
import lombok.Getter;
import lombok.NoArgsConstructor;
import java.math.BigInteger;
import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
public class BidResponse {
    private Long id;
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
