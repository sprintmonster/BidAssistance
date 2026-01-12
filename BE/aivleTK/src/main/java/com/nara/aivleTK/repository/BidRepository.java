package com.nara.aivleTK.repository;

import com.nara.aivleTK.domain.Bid;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface BidRepository extends JpaRepository<Bid,Long> {
    boolean existsByBidRealId(String realId);
    List<Bid> findByNameContainingOrOrganizationContainingOrRegionContaining(String name, String organization,String region);
    List<Bid> findByBidRealIdIn(List<String> realIds);
    List<Bid> findTop200ByRegionIsNull();

}
