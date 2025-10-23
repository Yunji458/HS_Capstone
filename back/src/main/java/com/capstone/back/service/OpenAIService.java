package com.capstone.back.service;

import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.List;

@Service
public class OpenAIService {

    @Value("${openai.api.key}")
    private String openaiApikey;

    @Value("${openai.api.url}")
    private String openaiApiUrl;

    @Value("${openai.model}")
    private String openaiModel;

    public List<String> extractWords(String sentence) {
        try {
            RestTemplate restTemplate = new RestTemplate();
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.setBearerAuth(openaiApikey);

            String prompt = String.format("""
                다음 문장은 수어로 번역될 예정입니다. 이 문장에서 수어 영상과 매핑할 수 있도록 핵심 의미 단어만 추출해주세요. 
                조사, 접속사, 불필요한 보조어구는 제외하고, 수어로 표현 가능한 의미 있는 단어만 남기세요.
                결과는 JSON 배열 형식으로 출력하세요. 형식: ["단어1", "단어2", ...]
                문장: "%s"
            """, sentence);

            JSONObject message = new JSONObject();
            message.put("role", "user");
            message.put("content", prompt);

            JSONObject requestBody = new JSONObject();
            requestBody.put("model", openaiModel);
            requestBody.put("messages", new org.json.JSONArray().put(message)); // ✅ 복수형
            requestBody.put("temperature", 0.4);

            HttpEntity<String> entity = new HttpEntity<>(requestBody.toString(), headers);

            ResponseEntity<String> response = restTemplate.postForEntity(openaiApiUrl, entity, String.class);

            if (response.getStatusCode() == HttpStatus.OK) {
                JSONObject json = new JSONObject(response.getBody());
                String content = json
                        .getJSONArray("choices")
                        .getJSONObject(0)
                        .getJSONObject("message")
                        .getString("content")
                        .replaceAll("[\\n\\r]", "")
                        .trim();

                org.json.JSONArray array = new org.json.JSONArray(content);
                List<String> words = new ArrayList<>();
                for (int i = 0; i < array.length(); i++) {
                    words.add(array.getString(i));
                }
                return words;
            } else {
                throw new RuntimeException("OpenAI API 호출 실패: " + response.getStatusCode());
            }
        } catch (Exception e) {
            throw new RuntimeException("OpenAI 처리 중 에러 발생", e);
        }
    }
}
