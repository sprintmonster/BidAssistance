package com.nara.aivleTK.service;

import com.nara.aivleTK.domain.company.Company;
import com.nara.aivleTK.dto.company.CompanyRequest;
import com.nara.aivleTK.dto.company.CompanyResponse;
import com.nara.aivleTK.exception.ResourceNotFoundException;
import com.nara.aivleTK.repository.CompanyRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class CompanyServiceImpl implements CompanyService {

    private final CompanyRepository companyRepository;

    // 회사 생성
    @Override
    @Transactional
    public CompanyResponse createCompany(CompanyRequest request) {
        companyRepository.findByName(request.getName())
                .ifPresent(c -> {
                    throw new IllegalStateException("이미 등록된 이름");
                });
        Company company = Company.builder()
                .name(request.getName())
                .license(request.getLicense())
                .performanceHistory(request.getPerformanceHistory())
                .build();
        return CompanyResponse.from(companyRepository.save(company));
    }

    // 회사 조회
    @Override
    public CompanyResponse getCompany(Integer id) {
        Company company = companyRepository.findById(id)
                .orElseThrow(()->new ResourceNotFoundException("회사를 찾을 수 없습니다."));

        return CompanyResponse.from(company);
    }

    // 회사 목록 모두 조회
    @Override
    public List<CompanyResponse> getAllCompanies() {
        return companyRepository.findAll().stream().map(CompanyResponse::from).collect(Collectors.toList());
    }

    // 회사 정보 수정
    @Override
    @Transactional
    public CompanyResponse updateprofile(Integer id, String license, String performanceHistory) {
        Company company = companyRepository.findById(id)
                .orElseThrow(()->new ResourceNotFoundException("회사를 찾을 수 없습니다."));
        company.setLicense(license);
        company.setPerformanceHistory(performanceHistory);
        return CompanyResponse.from(company);
    }
}