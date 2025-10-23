package com.capstone.back.repository;

import com.capstone.back.entity.SignVideo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface SignVideoRepository extends JpaRepository<SignVideo, Long> {
    Optional<SignVideo> findByName(String name);

    @Query("SELECT s.name FROM SignVideo s WHERE s.displayName = :displayName")
    String findNameByDisplayName(@Param("displayName") String displayName);
}

