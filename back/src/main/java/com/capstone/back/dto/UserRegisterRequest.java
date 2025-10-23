package com.capstone.back.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class UserRegisterRequest {
    private String userId;
    private String password;
    private String nickname;
    private String studentId;
}
