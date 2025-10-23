package com.capstone.back.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class UserScoreDto {
    private String nickname;
    private String imageBase64;
    private Integer sum;
}
