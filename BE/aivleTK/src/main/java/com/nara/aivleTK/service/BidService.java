package com.nara.aivleTK.service;
import com.nara.aivleTK.dto.BidResponse;
import java.util.List;

public interface BidService {
    List<BidResponse> searchBid(String keyword);
    //나중에 투찰 마감시간이 지난 공고는 제외 할 수 있도록 변경
    List<BidResponse> getAllBid();
    BidResponse getBidById(long id);


}
