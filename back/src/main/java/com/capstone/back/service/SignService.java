package com.capstone.back.service;

import com.capstone.back.repository.SignVideoRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;


@Service
public class SignService {

    @Autowired
    private OpenAIService openAIService;

    @Autowired
    private SignVideoRepository signVideoRepository;

    public List<String> getVideoUrlsFromSentence(String sentence) {
        List<String> keywords = openAIService.extractWords(sentence);
        List<String> urls = new ArrayList<>();

        for (String keyword : keywords) {
            signVideoRepository.findByName(keyword).ifPresent(signVideo -> {
                String videoUrl = "/videos/" + signVideo.getDisplayName() + ".mp4";
                urls.add(videoUrl);
            });
        }

        return urls;
    }
}
