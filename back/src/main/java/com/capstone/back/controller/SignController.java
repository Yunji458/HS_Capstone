package com.capstone.back.controller;

import com.capstone.back.service.SignService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/sign")
public class SignController {

    @Autowired
    private SignService signService;

    @PostMapping("/videos")
    public List<String> getSignVideos(@RequestBody String sentence) {
        return signService.getVideoUrlsFromSentence(sentence);
    }
}