package com.capstone.back.controller;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.capstone.back.entity.SignVideo;
import com.capstone.back.repository.SignVideoRepository;

@RestController
@RequestMapping("/api")
public class QuizController {
    
    @Autowired
    private SignVideoRepository signVideoRepository;

    @GetMapping("/quiz")
    public ResponseEntity<Map<String, String>> getRandomSignVideo() {
        List<SignVideo> videos = signVideoRepository.findAll();
        if(videos.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Collections.emptyMap());
        }

        // 랜덤 선택
        SignVideo randomVideo = videos.get(new Random().nextInt(videos.size()));

        Map<String, String> response = new HashMap<>();
        response.put("word", randomVideo.getName());

        return ResponseEntity.ok(response);
    }

    // 세미 백엔드 추가
    @GetMapping("/quiz/test")
    public ResponseEntity<List<String>> testFindAll() {
        List<SignVideo> all = signVideoRepository.findAll();
        List<String> names = all.stream().map(SignVideo::getName).toList();
        return ResponseEntity.ok(names);
    }

}
