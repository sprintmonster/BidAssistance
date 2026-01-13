package com.nara.aivleTK.controller;





import com.nara.aivleTK.service.AnalysisService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/analysis")
public class AnalysisResultController {
    private final AnalysisService analysisService;

    // 1. 분석 요청 API (프론트에서 '분석하기' 버튼 클릭 시 호출)
    // POST /api/analysis/predict/10 (공고 ID 10번 분석 요청)
    @PostMapping("/predict/{bidId}")
    public ResponseEntity<String> performAnalysis(@PathVariable Integer bidId) {
        try {
            analysisService.analyzeAndSave(bidId);
            return ResponseEntity.ok("분석이 완료되었습니다.");
        } catch (Exception e) {
            return ResponseEntity.status(500).body("분석 중 오류 발생: " + e.getMessage());
        }
    }

}
