package com.capstone.back.repository;

import com.capstone.back.domain.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    boolean existsByUserId(String userId);

    User findByUserIdAndPassword(String userId, String password);
    User findByUserId(String userId);

    List<User> findTop10ByOrderBySumDesc();
}
