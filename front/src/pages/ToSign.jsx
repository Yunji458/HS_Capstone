import { useState } from "react";
import { motion } from "framer-motion";
import axios from "axios";

export default function SignVideoPlayer() {
  const [word, setWord] = useState("");
  const [videoUrls, setVideoUrls] = useState([]);
  const [submitted, setSubmitted] = useState(false);

  const BACKEND_URL = "http://localhost:8080";

  const handleSearch = async () => {
    if (!word.trim()) return;
    try {
      const res = await axios.post(`${BACKEND_URL}/api/sign/videos`, word, {
        headers: { "Content-Type": "text/plain" },
      });

      console.log("🔍 OpenAI + DB 결과 영상 경로 목록:", res.data);

      // ✔ 절대경로로 바꾸기
      const fullUrls = res.data.map(url => `${BACKEND_URL}${url}`);
      setVideoUrls(fullUrls);
      setSubmitted(true);
    } catch (err) {
      alert("에러 발생: " + err.message);
    }
  };

  return (
    <div
      style={{
        backgroundColor: "#fffff2",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: submitted ? "flex-start" : "center",
        paddingTop: submitted ? "5rem" : "0",
        transition: "all 0.5s ease",
      }}
    >
      <motion.h2
        initial={{ opacity: 0, y: 0 }}
        animate={{ opacity: 1, y: submitted ? -30 : 0 }}
        transition={{ duration: 1.6, ease: "easeInOut" }}
        style={{
          marginBottom: "3rem",
          fontSize: "28px",
          fontWeight: "bold",
          color: "#3c4b3f",
        }}
      >
        문장을 수어로 번역
      </motion.h2>

      <motion.div
        initial={{ y: 0 }}
        animate={{ opacity: 1, y: submitted ? -30 : 0 }}
        transition={{ duration: 1.6, ease: "easeInOut" }}
        style={{
          marginBottom: submitted ? "2rem" : "0",
          display: "flex",
          gap: "10px",
        }}
      >
        <input
          type="text"
          placeholder="문장을 입력하세요."
          value={word}
          onChange={(e) => setWord(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSearch();
          }}
          style={{
            width: "720px",
            padding: "12px 16px",
            fontSize: "18px",
            borderRadius: "8px",
            border: "1px solid #ccc",
            boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
          }}
        />
        <button
          onClick={handleSearch}
          style={{
            padding: "12px 20px",
            fontSize: "16px",
            backgroundColor: "#7e9e89",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          확인
        </button>
      </motion.div>

      {submitted && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 60 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 2, ease: "easeInOut" }}
          style={{
            marginTop: "1.5rem",
            backgroundColor: "#fff",
            padding: "2rem",
            borderRadius: "16px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
            textAlign: "center",
          }}
        >
          <h3 style={{ marginBottom: "1rem", fontSize: "20px", fontWeight: "bold" }}>
            수어 영상
          </h3>
          {videoUrls.map((url, idx) => (

            <video
              key={idx}
              style={{ width: "100%", maxWidth: "960px", marginBottom: "1rem", borderRadius: "12px" }}
              controls
              autoPlay
              muted
              playsInline
              onError={(e) => {
                console.error("⚠️ 비디오 로딩 실패:", url, e);
              }}
            >
              <source src={url} type="video/mp4" />
              비디오를 불러올 수 없습니다.
            </video>

          ))}
        </motion.div>
      )}
    </div>
  );
}
