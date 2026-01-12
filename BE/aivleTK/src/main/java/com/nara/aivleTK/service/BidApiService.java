package com.nara.aivleTK.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nara.aivleTK.domain.Bid;
import com.nara.aivleTK.dto.BidApiDto;
import com.nara.aivleTK.repository.BidRepository;
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
    private final String SERVICE_KEY = "c1588436fef59fe2109d0eb3bd03747f61c57a482a6d0052de14f85b0bb02fb2";

    public String fetchAndSaveBidData() {
        try {
            // 1. [목록 API 호출] (기존과 동일)
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

            // 2. [중복 제거] (DB에 없는 것만 남김)
            List<String> realIdsToCheck = fetchedBids.stream().map(Bid::getBidRealId).collect(Collectors.toList());
            List<Bid> existingBids = bidRepository.findByBidRealIdIn(realIdsToCheck);
            Set<String> existingIds = existingBids.stream().map(Bid::getBidRealId).collect(Collectors.toSet());

            List<Bid> newBidsToSave = fetchedBids.stream()
                    .filter(bid -> !existingIds.contains(bid.getBidRealId()))
                    .collect(Collectors.toList());

            // ★ 3. [데이터 병합] 참가가능지역 API 호출하여 정보 합치기 ★
            for (Bid bid : newBidsToSave) {
                try {
                    // 공고번호를 넘겨서 참가가능지역(문자열)을 받아옴
                    String permittedRegion = getPermittedRegion(bid.getBidRealId());

                    // 받아온 값을 객체에 세팅 (Merge)
                    bid.setRegion(permittedRegion);

                    // API 서버 보호용 딜레이 (0.05초)
                    Thread.sleep(50);

                } catch (Exception e) {
                    log.error("지역정보 병합 실패 (ID: {}): {}", bid.getBidRealId(), e.getMessage());
                    bid.setRegion("확인필요");
                }
            }

            // 4. [최종 저장] (지역정보가 포함된 상태로 Insert)
            if (!newBidsToSave.isEmpty()) {
                bidRepository.saveAll(newBidsToSave);
            }

            return "신규 " + newBidsToSave.size() + "건 저장 완료 (지역제한 정보 포함)";

        } catch (Exception e) {
            log.error("Error", e);
            return "에러: " + e.getMessage();
        }
    }

    // ★ [Helper] 참가가능지역 정보 조회 API (getBidPblancListInfoPrtcptPsblRgn)
// [디버깅 모드] 참가가능지역 API 호출
    // [최종 해결] 공고번호와 차수를 분리하여 전송하는 코드
    // [최종_진짜_완료] 사용자 제공 URL 패턴 적용 (inqryDiv=2 추가)
    private String getPermittedRegion(String fullBidNtceNo) {
        // 1. 공고번호(bidNtceNo)와 차수(bidNtceOrd) 분리
        String baseNo = fullBidNtceNo;
        String ord = "00"; // 기본값

        // "R25BK01205208-000" 형태에서 분리
        if (fullBidNtceNo.contains("-")) {
            String[] parts = fullBidNtceNo.split("-");
            baseNo = parts[0];
            if (parts.length > 1) {
                ord = parts[1];
            }
        }

        java.net.HttpURLConnection conn = null;
        try {
            // 2. URL 생성 (사용자분이 주신 '진짜 되는 주소' 패턴 적용)
            StringBuilder urlBuilder = new StringBuilder("https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoPrtcptPsblRgn");
            urlBuilder.append("?" + URLEncoder.encode("serviceKey", "UTF-8") + "=" + SERVICE_KEY);
            urlBuilder.append("&" + URLEncoder.encode("pageNo", "UTF-8") + "=" + URLEncoder.encode("1", "UTF-8"));
            urlBuilder.append("&" + URLEncoder.encode("numOfRows", "UTF-8") + "=" + URLEncoder.encode("10", "UTF-8"));
            urlBuilder.append("&" + URLEncoder.encode("type", "UTF-8") + "=" + URLEncoder.encode("json", "UTF-8"));

            // ★ 핵심 추가: inqryDiv=2
            urlBuilder.append("&" + URLEncoder.encode("inqryDiv", "UTF-8") + "=" + URLEncoder.encode("2", "UTF-8"));

            // 공고번호와 차수
            urlBuilder.append("&" + URLEncoder.encode("bidNtceNo", "UTF-8") + "=" + URLEncoder.encode(baseNo, "UTF-8"));
            urlBuilder.append("&" + URLEncoder.encode("bidNtceOrd", "UTF-8") + "=" + URLEncoder.encode(ord, "UTF-8"));

            URL url = new URI(urlBuilder.toString()).toURL();
            conn = (java.net.HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setRequestProperty("Content-type", "application/json");

            // 3. User-Agent 헤더 (봇 차단 방지)
            conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36");

            conn.setConnectTimeout(5000);
            conn.setReadTimeout(5000);

            // 4. 응답 읽기 (NPE 방지)
            int responseCode = conn.getResponseCode();
            java.io.BufferedReader rd;

            if(responseCode >= 200 && responseCode <= 300) {
                rd = new java.io.BufferedReader(new java.io.InputStreamReader(conn.getInputStream(), "UTF-8"));
            } else {
                java.io.InputStream errStream = conn.getErrorStream();
                if (errStream == null) {
                    log.warn("⛔ API 응답 없음 (Code: {}). ID: {} -> 전국 처리", responseCode, fullBidNtceNo);
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

            // 5. XML 에러 체크
            if (responseStr.trim().startsWith("<")) {
                // 필요시 로그 주석 해제하여 확인
                // log.warn("API XML 반환(에러): {}", responseStr);
                return "전국";
            }

            // 6. JSON 파싱
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
            log.error("지역 조회 중 예외 (ID: {}): {}", fullBidNtceNo, e.getMessage());
            return "전국";
        } finally {
            if (conn != null) conn.disconnect();
        }
    }
}