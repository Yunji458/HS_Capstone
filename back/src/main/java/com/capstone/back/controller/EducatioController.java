package com.capstone.back.controller;

import com.capstone.back.entity.SignVideo;
import com.capstone.back.service.EducationService;
import org.springframework.web.bind.annotation.*;

import java.util.List;
@RestController
@RequestMapping("/api/education")
@CrossOrigin(origins = "http://localhost:3000")

public class EducatioController {

    private final EducationService educationService;

    public EducatioController(EducationService educationService) {
        this.educationService = educationService;
    }

    @GetMapping
    public List<SignVideo> getAll() {
        return educationService.getAllVideos();
    }

    @GetMapping("/search")
    public List<SignVideo> searchByName(@RequestParam String name) {
        return educationService.searchByName(name);
    }

    @GetMapping("/category")
    public List<SignVideo> filerByCategory(@RequestParam String category) {
        return educationService.filterByCategory(category);
    }

    @GetMapping("/all")
    public List<SignVideo> getAlledu() {
        return educationService.getAllVideos();
    }
}
