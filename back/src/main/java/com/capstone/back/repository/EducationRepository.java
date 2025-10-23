package com.capstone.back.repository;

import com.capstone.back.entity.SignVideo;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface EducationRepository extends JpaRepository<SignVideo, Long> {
    //  이름 기반 검색
    List<SignVideo> findByNameContainingIgnoreCase(String name);

    // 카테고리 필터링
    List<SignVideo> findByCategory(String category);
}
