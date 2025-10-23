package com.capstone.back.controller;

import com.capstone.back.dto.CollectRequest;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/api")
public class CollectController {

    @Autowired
    private RestTemplate restTemplate;

    @PostMapping("/collect")
    public ResponseEntity<String> collect(@RequestBody CollectRequest request) {
        int frames = request.getSequence() != null ? request.getSequence().size() : 0;
        int dims = frames > 0 ? request.getSequence().get(0).size() : 0;
        System.out.println("Spring Boot: 받은 좌표 시퀀스 frames=" + frames + ", dims=" + dims);
        System.out.println("Spring Boot: 받은 action=" + request.getAction());

        String flaskUrl = "http://localhost:5000/collect";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        try {
            ObjectMapper mapper = new ObjectMapper();
            String json = String.format("{\"label\":\"%s\", \"sequence\": %s}",
                    request.getAction(),
                    mapper.writeValueAsString(request.getSequence()));

            HttpEntity<String> entity = new HttpEntity<>(json, headers);
            ResponseEntity<String> response = restTemplate.postForEntity(flaskUrl, entity, String.class);
            return new ResponseEntity<>(response.getBody(), response.getStatusCode());
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("JSON 변환 중 오류 발생: " + e.getMessage());
        }
    }
}
