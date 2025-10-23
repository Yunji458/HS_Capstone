package com.capstone.back.controller;

import com.capstone.back.dto.SequenceRequest;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/api")
public class SpellingQuizController {
    @Autowired
    private RestTemplate restTemplate;

    @PostMapping("/perceive")
    public ResponseEntity<String> predict(@RequestBody SequenceRequest req) {
        String flaskUrl = "http://localhost:5000/predict";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<SequenceRequest> entity = new HttpEntity<>(req, headers);

        // 파이썬 Flask 서버에 POST 요청 보내서 결과를 받음
        ResponseEntity<String> resp = restTemplate.postForEntity(flaskUrl, entity, String.class);

        // Flask 서버가 준 응답 본문과 상태 코드를 그대로 클라이언트에 전달
        return new ResponseEntity<>(resp.getBody(), resp.getStatusCode());
    }
}
