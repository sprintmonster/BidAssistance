package com.nara.aivleTK.service;

import com.nara.aivleTK.domain.AnalysisResult;
import com.nara.aivleTK.domain.Bid;
import com.nara.aivleTK.dto.AnalysisResultDto;
import com.nara.aivleTK.repository.AnalysisResultRepository;
import com.nara.aivleTK.repository.BidRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.reactive.function.client.WebClient;

@Service
@RequiredArgsConstructor
public class AnalysisService {
    private final AnalysisResultRepository analysisResultRepository;
    private final BidRepository bidRepository;
    private final WebClient webClient;

    @Transactional
    @Async
    public void analyzeAndSave(Integer bidId){
        Bid sourceData = bidRepository.findById(bidId)
                .orElseThrow(()-> new IllegalArgumentException("Invalid bid ID: " + bidId));
        AnalysisResultDto response = webClient.post()
                .uri("/predict")
                .bodyValue(sourceData)
                .retrieve()
                .bodyToMono(AnalysisResultDto.class)
                .block();
        if(response==null){
            throw new IllegalArgumentException("AI 서버로 부터 응답없음");
        }
        AnalysisResult entity = new AnalysisResult();
        entity.setBidBidId(response.getBidBidId());
        entity.setGoldenRate(response.getGoldenRate());
        entity.setPredictedPrice(response.getPredictPrice());
        entity.setAvgRate(response.getAvgRate());
        entity.setFilepath(response.getFilepath());
        entity.setAnalysisContent(response.getAnalysisContent());

        analysisResultRepository.save(entity);
    }
}
