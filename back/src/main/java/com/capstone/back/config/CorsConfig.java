package com.capstone.back.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.*;
import org.springframework.lang.NonNull;

@Configuration
public class CorsConfig implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(@NonNull CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOrigins("http://localhost:3000")  // 프론트 주소
                .allowedMethods("*")
                .allowedHeaders("*")
                .allowCredentials(true); // ✅ 세션 쿠키 허용
    }

    // 세미 백엔드 추가
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/videos/**")
                .addResourceLocations("file:/Users/semi/HSwebFile/videos/") // 실제 video 경로로 수정할것
                .setCachePeriod(3600)
                .resourceChain(true);
    }
}
