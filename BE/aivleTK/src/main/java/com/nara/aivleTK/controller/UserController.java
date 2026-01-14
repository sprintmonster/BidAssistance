package com.nara.aivleTK.controller;

import com.nara.aivleTK.dto.user.LoginRequest;
import com.nara.aivleTK.dto.user.ResetPasswordRequest;
import com.nara.aivleTK.dto.user.UserCreateRequest;
import com.nara.aivleTK.dto.user.UserResponse;
import com.nara.aivleTK.service.UserService;
import jakarta.servlet.http.HttpSession;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    // 1. 유저 생성
    @PostMapping
    public ResponseEntity<UserResponse> createUser(@RequestBody UserCreateRequest user) {
        UserResponse saved = userService.createUser(user);
        return ResponseEntity.status(HttpStatus.CREATED).body(saved);
    }

    // 2. 유저 조회
    @GetMapping("/{id}")
    public ResponseEntity<UserResponse> getUser(@PathVariable("id") Integer id) {
        UserResponse userResponse = userService.getUserInfo(id);
        return ResponseEntity.ok(userResponse);
    }

    // 3. 로그인 (POST)
    @PostMapping("/login")
    public ResponseEntity<UserResponse> login(@RequestBody LoginRequest request, HttpSession session) {
        UserResponse loginUser = userService.login(request);

        // 세션에 회원 정보 저장 (서버가 "이 사람 로그인했음" 하고 기억표를 줌)
        session.setAttribute("loginUser", loginUser);

        return ResponseEntity.ok(loginUser);
    }

    // 4. 로그아웃 (POST)
    @PostMapping("/logout")
    public ResponseEntity<String> logout(HttpSession session) {
        session.invalidate(); // 세션 폭파 (기억표 삭제)
        return ResponseEntity.ok("로그아웃 되었습니다.");
    }

    // 5. 아이디 찾기 (GET)
    @GetMapping("/find_email")
    public ResponseEntity<Object> findEmail(@RequestParam String name, @RequestParam String answer, @RequestParam LocalDate birth) {
        String email = userService.findEmail(name, answer, birth);
        return ResponseEntity.ok(email);
    }

    // 6. 비밀번호 찾기 (GET)
    @PostMapping("/reset_password/")
    public ResponseEntity<Object> resetPassword(@RequestBody ResetPasswordRequest rpr) {
        String password = userService.resetPassword(rpr.getEmail(), rpr.getName(), rpr.getAnswer(), rpr.getBirth());
        return ResponseEntity.ok("임시 비밀번호가 발급 되었습니다.");
    }

    // 7. 회원정보 수정 (PUT)
    @PutMapping("/{id}")
    public ResponseEntity<UserResponse> updateUser(@PathVariable Integer id, @RequestBody UserCreateRequest request) {
        UserResponse updatedUser = userService.updateUser(id, request);
        return ResponseEntity.ok(updatedUser);
    }

    // 8. 회원정보 삭제 (DELETE)
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(@PathVariable Integer id) {
        userService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }

    // 9. 휴먼 계정 전환
    @PostMapping("/restUser/{id}")
    public ResponseEntity<Void> restUser(@PathVariable Integer id, @RequestParam Integer rest) { // rest가 3이면 휴면 계정 전환, 그 외면 role의 해당하는 숫자로 전환
        userService.restUser(id, rest);
        return ResponseEntity.noContent().build();
    }

    // 10. 로그인 확인
    @GetMapping("/checkLogin")
    public ResponseEntity<UserResponse> checkLogin(HttpSession session) {
        UserResponse loginUser = (UserResponse) session.getAttribute("loginUser");

        if (loginUser == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }

        return ResponseEntity.ok(loginUser);
    }
}