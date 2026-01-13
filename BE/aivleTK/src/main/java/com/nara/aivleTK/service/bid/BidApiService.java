package com.nara.aivleTK.service.bid;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nara.aivleTK.domain.Bid;
import com.nara.aivleTK.dto.bid.BidApiDto;
import com.nara.aivleTK.repository.BidRepository;
import com.nara.aivleTK.service.AnalysisService; // ★ 추가됨
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.URL;
import java.net.URLEncoder;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class BidApiService {

    private final BidRepository bidRepository;
    private final AnalysisService analysisService; // ★ 추가됨: AI 분석 서비스 주입
    private final String SERVICE_KEY = "c1588436fef59fe2109d0eb3bd03747f61c57a482a6d0052de14f85b0bb02fb2";

    public String fetchAndSaveBidData() {
        try {
            // 1. [목록 API 호출]
            LocalDateTime now = LocalDateTime.now();
            LocalDateTime start = now.minusHours(12);
            LocalDateTime end = now.plusHours(12);
            DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyyMMddHHmm");

            StringBuilder listUrlBuilder = new StringBuilder("http://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoCnstwk");
            listUrlBuilder.append("?" + URLEncoder.encode("serviceKey", "UTF-8") + "=" + SERVICE_KEY);
            listUrlBuilder.append("&" + URLEncoder.encode("numOfRows", "UTF-8") + "=" + URLEncoder.encode("200", "UTF-8"));
            listUrlBuilder.append("&" + URLEncoder.encode("pageNo", "UTF-8") + "=" + URLEncoder.encode("1", "UTF-8"));
            listUrlBuilder.append("&" + URLEncoder.encode("inqryDiv", "UTF-8") + "=" + URLEncoder.encode("1", "UTF-8"));
            listUrlBuilder.append("&" + URLEncoder.encode("inqryBgnDt", "UTF-8") + "=" + URLEncoder.encode(start.format(fmt), "UTF-8"));
            listUrlBuilder.append("&" + URLEncoder.encode("inqryEndDt", "UTF-8") + "=" + URLEncoder.encode(end.format(fmt), "UTF-8"));
            listUrlBuilder.append("&" + URLEncoder.encode("type", "UTF-8") + "=" + URLEncoder.encode("json", "UTF-8"));

            URL listUrl = new URI(listUrlBuilder.toString()).toURL();
            ObjectMapper mapper = new ObjectMapper();
            JsonNode rootNode = mapper.readTree(listUrl);
            JsonNode itemsNode = rootNode.path("response").path("body").path("items");

            if (itemsNode.isMissingNode() || itemsNode.isEmpty()) return "데이터 없음";

            List<Bid> fetchedBids = new ArrayList<>();
            if (itemsNode.isArray()) {
                for (JsonNode node : itemsNode) {
                    fetchedBids.add(mapper.treeToValue(node, BidApiDto.class).toEntity());
                }
            } else {
                fetchedBids.add(mapper.treeToValue(itemsNode.path("item"), BidApiDto.class).toEntity());
            }

            // 2. [중복 제거]
            List<String> realIdsToCheck = fetchedBids.stream().map(Bid::getBidRealId).collect(Collectors.toList());
            List<Bid> existingBids = bidRepository.findByBidRealIdIn(realIdsToCheck);
            Set<String> existingIds = existingBids.stream().map(Bid::getBidRealId).collect(Collectors.toSet());

            List<Bid> newBidsToSave = fetchedBids.stream()
                    .filter(bid -> !existingIds.contains(bid.getBidRealId()))
                    .collect(Collectors.toList());

            // 3. [데이터 병합] 참가가능지역 API 호출
            for (Bid bid : newBidsToSave) {
                try {
                    String permittedRegion = getPermittedRegion(bid.getBidRealId());
                    bid.setRegion(permittedRegion);
                    Thread.sleep(50); // API 서버 보호용 딜레이
                } catch (Exception e) {
                    log.error("지역정보 병합 실패 (ID: {}): {}", bid.getBidRealId(), e.getMessage());
                    bid.setRegion("확인필요");
                }
            }

            // ★ 4. [최종 저장 및 AI 분석 요청]
            if (!newBidsToSave.isEmpty()) {
                // (1) DB 저장 (ID 생성됨)
                List<Bid> savedBids = bidRepository.saveAll(newBidsToSave);

                // (2) AI 분석 요청 (비동기)
                int analysisCount = 0;
                for (Bid bid : savedBids) {
                    try {
                        analysisService.analyzeAndSave(bid.getBidId());
                        analysisCount++;
                    } catch (Exception e) {
                        log.error("AI 분석 요청 실패 (ID: {}): {}", bid.getBidId(), e.getMessage());
                    }
                }

                return "신규 " + savedBids.size() + "건 저장 완료, " + analysisCount + "건 분석 요청됨";
            }

            return "신규 데이터 없음";

        } catch (Exception e) {
            log.error("Error", e);
            return "에러: " + e.getMessage();
        }
    }

    // ★ 아래 메서드가 꼭 있어야 합니다!
    private String getPermittedRegion(String fullBidNtceNo) {
        String baseNo = fullBidNtceNo;
        String ord = "00";

        if (fullBidNtceNo.contains("-")) {
            String[] parts = fullBidNtceNo.split("-");
            baseNo = parts[0];
            if (parts.length > 1) {
                ord = parts[1];
            }
        }

        java.net.HttpURLConnection conn = null;
        try {
            StringBuilder urlBuilder = new StringBuilder("https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoPrtcptPsblRgn");
            urlBuilder.append("?" + URLEncoder.encode("serviceKey", "UTF-8") + "=" + SERVICE_KEY);
            urlBuilder.append("&" + URLEncoder.encode("pageNo", "UTF-8") + "=" + URLEncoder.encode("1", "UTF-8"));
            urlBuilder.append("&" + URLEncoder.encode("numOfRows", "UTF-8") + "=" + URLEncoder.encode("10", "UTF-8"));
            urlBuilder.append("&" + URLEncoder.encode("type", "UTF-8") + "=" + URLEncoder.encode("json", "UTF-8"));
            urlBuilder.append("&" + URLEncoder.encode("inqryDiv", "UTF-8") + "=" + URLEncoder.encode("2", "UTF-8"));
            urlBuilder.append("&" + URLEncoder.encode("bidNtceNo", "UTF-8") + "=" + URLEncoder.encode(baseNo, "UTF-8"));
            urlBuilder.append("&" + URLEncoder.encode("bidNtceOrd", "UTF-8") + "=" + URLEncoder.encode(ord, "UTF-8"));

            URL url = new URI(urlBuilder.toString()).toURL();
            conn = (java.net.HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setRequestProperty("Content-type", "application/json");
            conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36");
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(5000);

            int responseCode = conn.getResponseCode();
            java.io.BufferedReader rd;

            if (responseCode >= 200 && responseCode <= 300) {
                rd = new java.io.BufferedReader(new java.io.InputStreamReader(conn.getInputStream(), "UTF-8"));
            } else {
                java.io.InputStream errStream = conn.getErrorStream();
                if (errStream == null) {
                    return "전국";
                }
                rd = new java.io.BufferedReader(new java.io.InputStreamReader(errStream, "UTF-8"));
            }

            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = rd.readLine()) != null) {
                sb.append(line);
            }
            rd.close();

            String responseStr = sb.toString();

            if (responseStr.trim().startsWith("<")) {
                return "전국";
            }

            ObjectMapper mapper = new ObjectMapper();
            JsonNode rootNode = mapper.readTree(responseStr);
            JsonNode itemsNode = rootNode.path("response").path("body").path("items");

            if (itemsNode.isMissingNode() || itemsNode.isEmpty()) {
                return "전국";
            }

            List<String> regions = new ArrayList<>();
            if (itemsNode.isArray()) {
                for (JsonNode item : itemsNode) {
                    if (item.has("prtcptPsblRgnNm")) {
                        regions.add(item.get("prtcptPsblRgnNm").asText());
                    }
                }
            } else {
                if (itemsNode.has("item")) {
                    JsonNode innerItem = itemsNode.get("item");
                    if (innerItem.isArray()) {
                        for (JsonNode n : innerItem) {
                            if (n.has("prtcptPsblRgnNm")) regions.add(n.get("prtcptPsblRgnNm").asText());
                        }
                    } else if (innerItem.has("prtcptPsblRgnNm")) {
                        regions.add(innerItem.get("prtcptPsblRgnNm").asText());
                    }
                }
            }

            if (regions.isEmpty()) return "전국";
            return String.join(", ", regions);

        } catch (Exception e) {
            return "전국";
        } finally {
            if (conn != null) conn.disconnect();
        }
    }
}