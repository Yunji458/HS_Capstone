package com.capstone.back.controller;

import com.capstone.back.dto.FingerSpellRequestDto;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/fingerspell")
public class FingerSpellController {

    @PostMapping("/collect")
    public ResponseEntity<?> collectFingerSpell(@RequestBody FingerSpellRequestDto request) {
        String flaskUrl = "http://localhost:5001/collect";
        RestTemplate restTemplate = new RestTemplate();

        Map<String, Object> payload = new HashMap<>();
        payload.put("action", request.getAction());
        payload.put("sequence", request.getSequence());

        try {
            String result = restTemplate.postForObject(flaskUrl, payload, String.class);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.internalServerError().body("Flask 연동 실패: " + e.getMessage());
        }
    }
}
