package com.nara.aivleTK.dto.company;

import com.nara.aivleTK.domain.company.Company;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CompanyResponse {
    private Integer id;
    private String name;
    private String license;
    private String performanceHistory;

    public static CompanyResponse from(Company company) {
        return CompanyResponse.builder()
                .id(company.getId())
                .name(company.getName())
                .license(company.getLicense())
                .performanceHistory(company.getPerformanceHistory())
                .build();
    }
}
