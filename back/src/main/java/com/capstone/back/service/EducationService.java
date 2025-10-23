package com.capstone.back.service;

import java.util.List;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import com.capstone.back.entity.SignVideo;
import com.capstone.back.repository.EducationRepository;

@Service
public class EducationService {
    private final EducationRepository repository;

    public EducationService(EducationRepository repository) {
        this.repository = repository;
    }

    public List<SignVideo> getAllVideos() {
        return repository.findAll().stream()
                .map(this::enrich)
                .collect(Collectors.toList());
    }

    public List<SignVideo> searchByName(String name) {
        return repository.findByNameContainingIgnoreCase(name).stream()
                .map(this::enrich)
                .collect(Collectors.toList());
    }

    public List<SignVideo> filterByCategory(String category) {
        return repository.findByCategory(category).stream()
                .map(this::enrich)
                .collect(Collectors.toList());
    }

    private SignVideo enrich(SignVideo video) {
        String baseUrl = "http://localhost:8080";
        String fileKey = video.getDisplayName();
        video.setImageUrl(baseUrl + "/images/" + fileKey + ".png");
        video.setVideoUrl(baseUrl + "/videos/" + fileKey + ".mp4");
        return video;
    }
}
