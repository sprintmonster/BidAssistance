package com.nara.aivleTK.repository;

import com.nara.aivleTK.domain.user.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Integer> {

    Optional<User> findByEmail(String email); // 로그인

    boolean existsByEmail(String email);

    Optional<User> findAllByNameAndQuestionAndBirth(String name, String question, LocalDate birth);

    Optional<User> findByEmailAndNameAndQuestionAndBirth(String email, String name, String question, LocalDate birth);
}
