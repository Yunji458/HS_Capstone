package com.capstone.back.controller;

import com.capstone.back.dto.SequenceRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/api")
public class PredictionController {

    @Autowired
    private RestTemplate restTemplate;

    @PostMapping("/predict")
    public ResponseEntity<String> predict(@RequestBody SequenceRequest req) {
        // Flask /predict 로직 호출
        String flaskUrl = "http://localhost:5000/predict";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<SequenceRequest> entity = new HttpEntity<>(req, headers);
        ResponseEntity<String> resp = restTemplate.postForEntity(flaskUrl, entity, String.class);
        return new ResponseEntity<>(resp.getBody(), resp.getStatusCode());
    }
}
