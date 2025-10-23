package com.capstone.back.controller;

import com.capstone.back.domain.User;
import com.capstone.back.repository.UserRepository;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/user")
public class UserInfoController {
    
    @Autowired
    private UserRepository userRepository;

    @GetMapping("/me")
    public ResponseEntity<?> getCurrentUser(HttpSession session) {
        String userId = (String) session.getAttribute("userId");
        if(userId == null) {
            System.out.println("❌ 세션에 userId 없음");
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("세션 없음");
        }

        User user = userRepository.findByUserId(userId);
        if (user == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("사용자 없음");
        }
        return ResponseEntity.ok(user);
    }
}
