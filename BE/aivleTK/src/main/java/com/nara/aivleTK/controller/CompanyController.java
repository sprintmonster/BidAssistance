package com.nara.aivleTK.controller;

import com.nara.aivleTK.dto.company.CompanyRequest;
import com.nara.aivleTK.dto.company.CompanyResponse;
import com.nara.aivleTK.service.CompanyService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/company")
@RequiredArgsConstructor
public class CompanyController {
    private final CompanyService companyService;

    @PostMapping
    public ResponseEntity<CompanyResponse> createCompany(@RequestBody CompanyRequest cr) {
        CompanyResponse company = companyService.createCompany(cr);
        return ResponseEntity.status(HttpStatus.CREATED).body(company);
    }

    @GetMapping // 회사 전체 조회
    public ResponseEntity<List<CompanyResponse>> getALLCompanies() {
        List<CompanyResponse> list = companyService.getAllCompanies();
        return ResponseEntity.ok(list);
    }

    @GetMapping("/{id}") // 회사 상세 조회
    public ResponseEntity<CompanyResponse> getCompany(@PathVariable("id") Integer id) {
        CompanyResponse company = companyService.getCompany(id);
        return ResponseEntity.ok(company);
    }
}
