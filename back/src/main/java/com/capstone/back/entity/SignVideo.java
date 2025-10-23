package com.capstone.back.entity;

import com.fasterxml.jackson.annotation.JsonProperty;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Transient;
import lombok.Getter; 
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity
@Table(name = "sign_videos")
@Getter
@Setter
@NoArgsConstructor
public class SignVideo {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id; // 고유 ID

    @Column
    private String name; // 수어 이름

    @Column(name = "display_name")
    @JsonProperty("displayName")
    private String displayName; // 이미지 또는 영상 경로

    @Column
    private String category; // 단어 분류 (ex. 동물, 사물, 인사 등)

    @Transient
    private String imageUrl;

    @Transient
    private String videoUrl;
    public SignVideo( String name, String displayName, String category) {
        this.name = name;
        this.displayName = displayName;
        this.category = category;
    }
}
