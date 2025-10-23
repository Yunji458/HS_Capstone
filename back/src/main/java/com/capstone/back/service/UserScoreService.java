package com.capstone.back.service;

import com.capstone.back.domain.User;
import com.capstone.back.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import jakarta.transaction.Transactional;

@Service
public class UserScoreService {
    
    @Autowired
    private UserRepository userRepository;

    @Transactional
    public boolean updateScore(String userId, int score) {
        User user = userRepository.findByUserId(userId);
        
        if(user == null) {
            return false;
        }
        user.setScore(score);

        int signScore = user.getSignScore() != null ? user.getSignScore() : 0;
        user.setSum(score * 5 + signScore * 5);

        userRepository.save(user);
        return true;
    }

    @Transactional
    public boolean updateSignScore(String userId, int signScore) {
        User user = userRepository.findByUserId(userId);
        
        if(user == null) {
            return false;
        }
        user.setSignScore(signScore);

        int score = user.getScore() != null ? user.getScore() : 0;
        user.setSum(score * 5 + signScore * 5);

        userRepository.save(user);
        return true;
    }
}
