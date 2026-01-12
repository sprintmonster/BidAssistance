package com.nara.aivleTK.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HelloController {

    @GetMapping("/")
    public String home() {
        return "서버가 정상적으로 작동 중입니다! Hello World!";
    }
}