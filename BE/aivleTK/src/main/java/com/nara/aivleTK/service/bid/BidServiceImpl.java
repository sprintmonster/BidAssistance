package com.nara.aivleTK.service.bid;

import com.nara.aivleTK.domain.AnalysisResult;
import com.nara.aivleTK.domain.Bid;
import com.nara.aivleTK.dto.AnalysisResultDto;
import com.nara.aivleTK.dto.bid.BidResponse;
import com.nara.aivleTK.exception.ResourceNotFoundException;
import com.nara.aivleTK.repository.AnalysisResultRepository;
import com.nara.aivleTK.repository.BidRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class BidServiceImpl implements BidService {

    private final BidRepository bidRepository;
    private final AnalysisResultRepository analysisResultRepository;
    @Override
    public List<BidResponse> searchBid(String name, String region, String organization) {

        boolean noFilter =
                (name == null || name.isBlank()) &&
                        (region == null || region.isBlank()) &&
                        (organization == null || organization.isBlank());

        List<Bid> result = noFilter
                ? bidRepository.findAll()
                : bidRepository.findByNameContainingOrOrganizationContainingOrRegionContaining(
                name == null ? "" : name,
                organization == null ? "" : organization,
                region == null ? "" : region
        );

        return result.stream()
                .map(BidResponse::new)
                .toList();
    }

    @Override
    @Transactional(readOnly = true)
    public List<BidResponse> getAllBid() {
        return bidRepository.findAll()
                .stream()
                .map(BidResponse::new)
                .toList();
    }

    @Override
    @Transactional(readOnly = true)
    public BidResponse getBidById(int id) {

        Bid bid = bidRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Bid not Found. id =" + id));
        BidResponse response = new BidResponse(bid);
        analysisResultRepository.findByBidBidId(id).ifPresent(ar->
                response.setAnalysisResult(
                        AnalysisResultDto.builder()
                                .bidBidId(ar.getBid().getBidId())
                                .avgRate(ar.getAvgRate())
                                .goldenRate(ar.getGoldenRate())
                                .predictPrice(ar.getPredictedPrice())
                                .filepath(ar.getFilepath())
                                .analysisContent(ar.getAnalysisContent())
                                .build()
                )
        );
        return response;
    }
}
