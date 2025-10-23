package com.capstone.back.service;

import com.capstone.back.dto.UserScoreDto;
import com.capstone.back.repository.UserRepository;
import org.springframework.stereotype.Service;

import java.util.Base64;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public List<UserScoreDto> getTopUsers() {
        return userRepository.findTop10ByOrderBySumDesc()
                .stream()
                .map(user -> new UserScoreDto(
                        user.getNickname(),
                        user.getImage() != null ? "data:image/jpeg;base64," + Base64.getEncoder().encodeToString(user.getImage()) : null,
                        user.getSum()
                ))
                .collect(Collectors.toList());
    }
}