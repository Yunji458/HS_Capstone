// 세미 백엔드 수정
package com.capstone.back.controller;

import com.capstone.back.service.OpenAIService;
import com.capstone.back.service.SignVideoService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

// 2025 05 25 추가 - 비디오 랜덤으로 가져오기
import java.io.IOException;
import java.nio.file.*;

@RestController
@RequestMapping("/sign")
public class VideoController {
    @Autowired
    private SignVideoService signVideoService;
    @Autowired
    private OpenAIService openAIService;
    @GetMapping("/{word}")
    public ResponseEntity<String> getSignVideo(@PathVariable String word) {
        String fileName = signVideoService.getVideoFileName(word);

        if(fileName.equals("영상이 존재하지 않음")) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("영상 없음");
        }
        String videoUrl = "http://localhost:8080/videos" + fileName + ".mp4";
        return ResponseEntity.ok(videoUrl);
    }

    @PostMapping("/extract")
    public ResponseEntity<List<String>> extractWordsFromSentence(@RequestBody String sentence) {
        List<String> words = openAIService.extractWords(sentence);
        List<String> videoUrls = new ArrayList<>();
        for (String word : words) {
            String fileName = signVideoService.getVideoFileName(word);
            if(!fileName.equals("영상이 존재하지 않음")) {
                String videoUrl = "http://localhost:8080/videos/" + fileName + ".mp4";
                videoUrls.add(videoUrl);
            }
        }
        return ResponseEntity.ok(videoUrls);
    }

    // ✅ 랜덤 비디오 파일명 반환 엔드포인트 추가
    @GetMapping("/random")
    public ResponseEntity<Map<String, String>> getRandomVideo() throws IOException {
        Path videoDir = Paths.get("src/main/resources/static/videos");

        if (!Files.exists(videoDir)) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of("error", "Video directory not found."));
        }

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(videoDir, "*.mp4")) {
            List<Path> videoFiles = new ArrayList<>();
            for (Path path : stream) {
                videoFiles.add(path);
            }

            if (videoFiles.isEmpty()) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of("error", "No videos found."));
            }

            Path randomVideo = videoFiles.get(new Random().nextInt(videoFiles.size()));
            String fileName = randomVideo.getFileName().toString();
            String baseName = fileName.substring(0, fileName.lastIndexOf(".")); // display_name

            // 🧠 DB 조회해서 display_name → name 가져오기
            String name = signVideoService.getNameByDisplayName(baseName);  // 새로운 메서드 필요

            return ResponseEntity.ok(Map.of(
                "fileName", fileName,
                "baseName", baseName,
                "name", name // 사용자에게 보여질 한글 정답
            ));
        }
    }
}