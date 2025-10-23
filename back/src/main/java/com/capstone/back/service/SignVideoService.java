package com.capstone.back.service;

import com.capstone.back.entity.SignVideo;
import com.capstone.back.repository.SignVideoRepository;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class SignVideoService {
    @Autowired
    private SignVideoRepository signVideoRepository;

    private static final Logger logger = LoggerFactory.getLogger(SignVideoService.class);

    public String getDisplayName(String name) {
        logger.info("요청된 단어: " + name);

        return signVideoRepository.findByName(name)
                .map(video -> {
                    logger.info("영상 찾음: " + video.getDisplayName());
                    return video.getDisplayName();
                })
                .orElseGet(() -> {
                    logger.warn("영상 없음: " + name);
                    return "영상이 존재하지 않음";
                });
    }

    @GetMapping("/all")
    public List<SignVideo> getAllVideos() {
        return signVideoRepository.findAll().stream().map(video -> {
            String baseUrl = "http://localhost:8080";
            video.setVideoUrl(baseUrl + "/videos/" + video.getName() + ".mp4");
            return video;
        }).collect(Collectors.toList());
    }

    public String getVideoFileName(String name) {
        logger.info("요청된 단어: " + name);

        return signVideoRepository.findByName(name)
                .map(video -> {
                    logger.info("영상 찾음: " + video.getName());
                    return video.getName();
                })
                .orElseGet(() -> {
                    logger.warn("영상 없음: " + name);
                    return "영상이 존재하지 않음";
                });
    }

    public String getNameByDisplayName(String displayName) {
        return signVideoRepository.findNameByDisplayName(displayName);
    }
}
