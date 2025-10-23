package com.capstone.back.controller;

import com.capstone.back.domain.User;
import com.capstone.back.dto.UserRegisterRequest;
import com.capstone.back.dto.UserLoginRequest;
import com.capstone.back.dto.UserScoreDto;
import com.capstone.back.repository.UserRepository;
import com.capstone.back.service.UserService;

import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/* 사용자 정보 불러오기 */
import java.util.HashMap;
import java.util.Map;
import org.springframework.http.HttpStatus;
import java.util.List;

@RestController
@CrossOrigin(origins = "http://localhost:3000", allowCredentials = "true")
public class UserController {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private UserService userService;

    // 회원가입
    @PostMapping("/api/register")
    public ResponseEntity<String> register(@RequestBody UserRegisterRequest request) {
        if (userRepository.existsByUserId(request.getUserId())) {
            return ResponseEntity.ok("이미 존재하는 아이디입니다");
        }

        User user = new User();
        user.setUserId(request.getUserId());
        user.setPassword(request.getPassword());
        user.setNickname(request.getNickname());
        user.setStudentId(request.getStudentId());

        userRepository.save(user);
        return ResponseEntity.ok("회원가입 성공");
    }

    // 로그인
    @PostMapping("/api/login")
    public ResponseEntity<String> login(@RequestBody UserLoginRequest request, HttpSession session) {
        User user = userRepository.findByUserIdAndPassword(request.getUserId(), request.getPassword());

        System.out.println("request user ID: " + request.getUserId());
        System.out.println("request password: " + request.getPassword());

        if (user != null) {
            session.setAttribute("userId", user.getUserId());
            System.out.println("✅ session saved: " + session.getAttribute("userId"));
            System.out.println("✅ session ID: " + session.getId());
            return ResponseEntity.ok("로그인 성공");
        } else {
            return ResponseEntity.ok("로그인 실패");
        }
    }

    // 세션 확인
    @GetMapping("/api/session")
    public ResponseEntity<String> getSession(HttpSession session) {
        String userId = (String) session.getAttribute("userId");
        System.out.println("current session id: " + session.getId());
        System.out.println("session userId value: " + userId);

        if (userId != null) {
            return ResponseEntity.ok(userId);
        } else {
            return ResponseEntity.ok("세션 없음");
        }
    }

    // 로그아웃
    @PostMapping("/api/logout")
    public ResponseEntity<String> logout(HttpSession session) {
        session.invalidate();
        return ResponseEntity.ok("로그아웃 성공");
    }

    // 사용자 정보 불러오기
    @GetMapping("/api/userinfo")
    public ResponseEntity<?> getUserInfo(HttpSession session) {
        String userId = (String) session.getAttribute("userId");

        if (userId == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("세션 없음");
        }

        User user = userRepository.findByUserId(userId);
        if(user == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("사용자 없음");
        }

        Map<String, Object> userInfo = new HashMap<>();
        userInfo.put("userId", user.getUserId());
        userInfo.put("nickname", user.getNickname());
        userInfo.put("studentId", user.getStudentId());

        return ResponseEntity.ok(userInfo);
    }

    @GetMapping("/api/users/top")
    public ResponseEntity<List<UserScoreDto>> getTopUsers() {
        List<UserScoreDto> topUsers = userService.getTopUsers();
        return ResponseEntity.ok(topUsers);
    }
}
