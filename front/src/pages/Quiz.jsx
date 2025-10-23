import '../css/Quiz.css';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';

function Answer({ value, onChange, onEnter }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1, ease: "easeInOut" }}
    >
      <input
        type="text"
        placeholder="정답을 입력하세요"
        className="quiz-input"
        value={value}
        onChange={onChange}
        onKeyDown={(e) => {
          if (e.key === 'Enter') onEnter();
        }}
      />
    </motion.div>
  );
}

function SignView({ videoFileName }) {
  if (!videoFileName) return null;

  const fullPath = `http://localhost:8080/videos/${videoFileName}`;
  console.log("🎥 비디오 src 확인:", fullPath);

  return (
    <motion.div
      className="video-box"
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.7 }}
    >
      <video
        key={videoFileName}
        className='sign-video'
        muted
        autoPlay
        playsInline
        loop
      >
        <source src={fullPath} type="video/mp4" />
        브라우저가 비디오 태그를 지원하지 않습니다.
      </video>
    </motion.div>
  );
}

function Sidebar() {
  return (
    <div className="sidebar">
      <button><Link to="/quiz" className="link">수어 맞추기</Link></button>
      <button><Link to="/text-to-sign" className="link">지문자 맞추기</Link></button>
      <button><Link to="/scoreboard" className="link">스코어</Link></button>
    </div>
  );
}

export default function Quiz() {
  const [videoFileName, setVideoFileName] = useState("");
  const [word, setWord] = useState(""); // 정답 (한글)
  const [userInput, setUserInput] = useState('');
  const [score, setScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(60);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentUserId, setCurrentUserId] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8080/api/user/me", { credentials: "include" })
      .then(res => res.json())
      .then(data => {
        if (data?.userId) setCurrentUserId(data.userId);
      })
      .catch(err => console.error("❌ 사용자 정보 불러오기 실패:", err));
  }, []);

  const updateSignScore = async (score) => {
    if (!currentUserId) return;
    try {
      const res = await fetch("http://localhost:8080/api/user/updateSignScore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ userId: currentUserId, score })
      });
      const text = await res.text();
      console.log("✅ 수어 점수 저장 응답:", text);
    } catch (err) {
      console.error("❌ 수어 점수 저장 실패:", err);
    }
  };

  const loadNewQuiz = async () => {
    try {
      const response = await axios.get('http://localhost:8080/sign/random');
      const fileName = response.data.fileName;
      const name = response.data.name; // 한글 정답

      setVideoFileName(fileName);
      setWord(name);
      console.log("정답(개발자 콘솔 표시):", name);
    } catch (error) {
      console.error("비디오 로딩 실패:", error);
    }
  };

  useEffect(() => {
    let timer;
    if (isPlaying && timeLeft > 0) {
      timer = setInterval(() => {
        setTimeLeft(prev => prev - 1);
      }, 1000);
    } else if (timeLeft === 0 && isPlaying) {
      alert(`⏰ 시간이 종료되었습니다!\n맞춘 개수: ${score}개`);
      console.log(`게임 종료 - 맞춘 개수: ${score}개`);
      updateSignScore(score);
      setIsPlaying(false);
      setScore(0);
      setTimeLeft(60);
      setVideoFileName("");
    }
    return () => clearInterval(timer);
  }, [isPlaying, timeLeft]);

  const handleCheckAnswer = () => {
    if (userInput.trim() === word) {
      alert("정답입니다!");
      setScore(prev => prev + 1);
      setUserInput("");
      loadNewQuiz();
    } else {
      alert("틀렸습니다. 다시 시도해보세요!");
    }
  };

  const toggleGame = () => {
    if (!isPlaying) {
      setScore(0);
      setTimeLeft(60);
      setIsPlaying(true);
      loadNewQuiz();
      setTimeout(() => {
        window.scrollTo({ top: 150, behavior: 'smooth' });
      }, 100);
    } else {
      alert(`게임 종료!\n맞춘 개수: ${score}개`);
      console.log(`게임 종료 - 맞춘 개수: ${score}개`);
      updateSignScore(score);
      setIsPlaying(false);
      setScore(0);
      setTimeLeft(60);
      setVideoFileName("");
    }
  };

  return (
    <div className="quiz-page" style={{ display: "flex", backgroundColor: "#fffff2" }}>
      <Sidebar />
      <div className="quiz-container">
        <div style={{ width: "80%", maxWidth: "600px", marginBottom: "20px" }}>
          <motion.div
            initial={{ y: 0, opacity: 1 }}
            animate={{ y: -20 }}
            transition={{ duration: 1, ease: "easeInOut" }}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "20px",
              marginBottom: "1rem"
            }}
          >
            <motion.h1
              className='quiz-title'
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              수어 맞추기 게임 🎮
            </motion.h1>
            <motion.button
              onClick={toggleGame}
              className='quiz-button'
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6 }}
              style={{
                fontSize: "16px",
                padding: "10px 20px",
                cursor: "pointer",
                backgroundColor: isPlaying ? "#ff4d4d" : "#dcaf50",
                color: "#fff",
                border: "none",
                borderRadius: "6px"
              }}
            >
              {isPlaying ? "게임 종료" : "게임 시작"}
            </motion.button>
          </motion.div>
          <div style={{ marginBottom: "8px", fontWeight: "bold", textAlign: "center" }}>
            ⏱️ 남은 시간: {timeLeft}초 | 🏆 점수: {score}
          </div>
          <div style={{ height: "10px", backgroundColor: "#eee", borderRadius: "6px" }}>
            <motion.div
              style={{
                height: "100%",
                backgroundColor: "#e74c3c",
                borderRadius: "6px",
                transformOrigin: "left",
              }}
              animate={{ scaleX: timeLeft / 60 }}
              transition={{ duration: 0.2, ease: "linear" }}
            />
          </div>
        </div>

        <motion.div
          className="quiz-section"
          style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "30px" }}
        >
          <div className="quiz-video-large">
            <SignView videoFileName={videoFileName} />
          </div>
          <div className="quiz-answer-wide">
            <Answer
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onEnter={handleCheckAnswer}
            />
            <motion.button
              className="quiz-button"
              onClick={handleCheckAnswer}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              disabled={!isPlaying}
            >
              확인
            </motion.button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
