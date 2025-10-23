package com.capstone.back.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

@Entity
@Table(name = "user")
@Getter
@Setter
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String userId;

    @Column(nullable = false)
    private String password;

    @Column(nullable = false)
    private String nickname;

    @Column(nullable = false)
    private String studentId;

    @Column
    private Integer score = 0;

    @Column(name = "sign_score")
    private Integer signScore = 0;

    @Column(name = "sum")
    private Integer sum;

    @Lob
    @Column
    private byte[] image;
}
