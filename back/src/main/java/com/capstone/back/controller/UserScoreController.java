package com.capstone.back.controller;

import com.capstone.back.service.UserScoreService;
import lombok.Data;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/user")
public class UserScoreController {
    
    @Autowired
    private UserScoreService userScoreService;

    @PostMapping("/updateScore")
    public String updateUserScore(@RequestBody ScoreRequest request) {
        boolean updated = userScoreService.updateScore(request.getUserId(), request.getScore());
        return updated ? "✅ 점수 업데이트 완료" : "❌ 사용자 없음";
    }

    @PostMapping("/updateSignScore")
    public String updateSignScore(@RequestBody ScoreRequest request) {
        boolean updated = userScoreService.updateSignScore(request.getUserId(), request.getScore());
        return updated ? "✅ 수어 점수 업데이트 완료" : "❌ 사용자 없음";
    }

    @Data
    public static class ScoreRequest {
        private String userId;
        private int score;
    }
}
