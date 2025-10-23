package com.capstone.back.controller;

import org.springframework.web.bind.annotation.*;
import java.io.*;

@RestController
@RequestMapping("/api")
public class TrainingController {

    @PostMapping("/train")
    public String trainModel() {
        try {
            // 파이썬 스크립트 경로 지정 (상대경로 → 정규화된 절대경로로 출력)
            String scriptPath = "../lstmword/train_model.py";  // 백엔드 기준 상대경로

            File scriptFile = new File(scriptPath);
            String absoluteCanonicalPath = scriptFile.getCanonicalPath();
            System.out.println("🔹 실제 실행 경로: " + absoluteCanonicalPath);

            ProcessBuilder pb = new ProcessBuilder("python", absoluteCanonicalPath);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("[train_model.py] " + line);
                output.append(line).append("\n");
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("✅ Python 학습 완료");
                return "학습 완료!";
            } else {
                System.err.println("❌ 학습 실패 (Exit code: " + exitCode + ")");
                return "학습 실패: " + output;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return "학습 중 오류 발생: " + e.getMessage();
        }
    }
}
