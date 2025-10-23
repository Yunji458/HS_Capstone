import React, { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import "../css/Quiz.css";

const HANGUL_JAMOS = [
  'ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅌ','ㅍ','ㅎ',
  'ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅚ','ㅛ','ㅜ','ㅟ','ㅡ','ㅢ','ㅣ'
];

export default function FingerspellQuiz() {
  const videoRef = useRef(null);
  const holisticRef = useRef(null);
  const cameraRef = useRef(null);
  const collectingRef = useRef(false);
  const lastMatchedLetter = useRef('');
  const targetLetterRef = useRef('');
  const shouldSendToPython = useRef(false);
  const timerRef = useRef(null);

  const correctCountRef = useRef(0);

  const [sequence, setSequence] = useState([]);
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [isGameRunning, setIsGameRunning] = useState(false);
  const [targetLetter, setTargetLetter] = useState('');
  const [resultMessage, setResultMessage] = useState('');
  const [correctCount, setCorrectCount] = useState(0);
  const [timeLeft, setTimeLeft] = useState(60);
  
  const [currentUserId, setCurrentUserId] = useState(null);

  const updateUserScore = async (score) => {
    if (!currentUserId) {
      console.error("❌ currentUserId가 없습니다. 서버에서 사용자 정보를 불러오지 못했을 수 있습니다.");
      return;
    }

    try {
      const response = await fetch("http://localhost:8080/api/user/updateScore", {
        method: "POST",
        headers: { "Content-type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ userId: currentUserId, score })
      });
      const text = await response.text();
      console.log("✅ 서버 응답:", text);
    } catch (err) {
      console.log("❌ 점수 업데이트 실패:", err)
    }
  };

  const extractRawLeftHandLandmarks = (results) => {
    if (!results.leftHandLandmarks || results.leftHandLandmarks.length !== 21) return null;
    return results.leftHandLandmarks.flatMap(lm => [lm.x, lm.y, lm.z]);
  };

  function extractVectorFeatures3D_PythonEquivalent(landmarks_flat) {
    if (!landmarks_flat || landmarks_flat.length !== 21 * 3) return null;

    const getJoint = (i) => landmarks_flat.slice(i * 3, i * 3 + 3);
    const subtract = (a, b) => a.map((v, i) => v - b[i]);
    const norm = (v) => Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
    const normalize = (v) => {
      const n = norm(v);
      return n < 1e-6 ? [0, 0, 0] : v.map(val => val / n);
    };
    const dot = (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0);
    const toDeg = (rad) => rad * (180 / Math.PI);

    const wrist_landmark = getJoint(0);

    const v1_indices = [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19];
    const v2_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
    const bone_vectors_normalized = [];

    for (let i = 0; i < v1_indices.length; i++) {
      const joint1_coords = getJoint(v1_indices[i]);
      const joint2_coords = getJoint(v2_indices[i]);
      const v_orig_single = subtract(joint2_coords, joint1_coords);
      bone_vectors_normalized.push(...normalize(v_orig_single));
    }

    const get_bone_from_normalized_list = (i) => bone_vectors_normalized.slice(i * 3, i * 3 + 3);

    const angle_idx1 = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18];
    const angle_idx2 = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19];
    const angles_intra_finger = angle_idx1.map((idx1_val, k) => {
      const idx2_val = angle_idx2[k];
      const bone1 = get_bone_from_normalized_list(idx1_val);
      const bone2 = get_bone_from_normalized_list(idx2_val);
      let d = dot(bone1, bone2);
      d = Math.max(-1, Math.min(1, d));
      return toDeg(Math.acos(d));
    });

    const tip_indices = [4, 8, 12, 16, 20];
    const finger_tip_distances = [];

    const middle_mcp_landmark = getJoint(9);
    let reference_length = norm(subtract(middle_mcp_landmark, wrist_landmark));
    if (reference_length < 1e-5) {
        reference_length = 1.0;
    }

    for (let i = 0; i < tip_indices.length - 1; i++) {
        const tip1_coords = getJoint(tip_indices[i]);
        const tip2_coords = getJoint(tip_indices[i+1]);
        const distance = norm(subtract(tip1_coords, tip2_coords));
        finger_tip_distances.push(distance / reference_length);
    }

    const wrist_tip_angles = [];
    const vectors_wrist_to_tips_normalized = [];

    for (const tip_idx of tip_indices) {
        const tip_landmark_coords = getJoint(tip_idx);
        const vec_wrist_to_tip = subtract(tip_landmark_coords, wrist_landmark);
        vectors_wrist_to_tips_normalized.push(normalize(vec_wrist_to_tip));
    }

    for (let i = 0; i < vectors_wrist_to_tips_normalized.length - 1; i++) {
        const vec1 = vectors_wrist_to_tips_normalized[i];
        const vec2 = vectors_wrist_to_tips_normalized[i+1];
        let d = dot(vec1, vec2);
        d = Math.max(-1, Math.min(1, d));
        wrist_tip_angles.push(toDeg(Math.acos(d)));
    }

    return [
        ...bone_vectors_normalized,
        ...angles_intra_finger,
        ...finger_tip_distances,
        ...wrist_tip_angles
    ];
  }

  const getRandomLetter = () => HANGUL_JAMOS[Math.floor(Math.random() * HANGUL_JAMOS.length)];

  const logDebugInfo = (pred, count) => {
    console.log("🔤 현재 타겟 한글 자모:", targetLetterRef.current);
    console.log("🤟 사용자 수어 예측 자모:", pred);
    console.log("✅ 현재 맞춘 개수:", count);
  };

  const startGame = () => {
    setIsGameRunning(true);
    setCorrectCount(0);
    const first = getRandomLetter();
    setTargetLetter(first);
    targetLetterRef.current = first;
    setResultMessage('');
    lastMatchedLetter.current = '';
    setTimeLeft(60);
    shouldSendToPython.current = true;

    timerRef.current = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          clearInterval(timerRef.current);
          setResultMessage('⏰ 시간 종료!');
          stopGame();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    console.log("🎮 게임 시작됨. 첫 문제:", first);
  };

  const stopGame = () => {
    setIsGameRunning(false);
    shouldSendToPython.current = false;
    clearInterval(timerRef.current);
    setTargetLetter('');
    setResultMessage('');
    setTimeLeft(60);
    lastMatchedLetter.current = '';

    const finalScore = correctCountRef.current;
    alert(`게임종료! 맞춘 개수: ${correctCountRef.current}`);
    updateUserScore(finalScore);
  };

  useEffect(() => {
    fetch("http://localhost:8080/api/user/me", { credentials: "include" })
      .then(async res => {
        const contentType = res.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
          const data = await res.json();
          console.log("서버 응답 데이터:", data);
          if (data?.userId) {
            setCurrentUserId(data.userId);
            console.log("✅ 로그인된 사용자:", data.userId);
          } else {
            console.error("❌ 로그인된 사용자 정보 없음");
          }
        } else {
          const text = await res.text();
          console.error("❌ JSON 아님 / 상태 코드:", res.status, text);
        }
      })
      .catch(err => console.error("❌ 사용자 정보 불러오기 실패:", err));
  }, []);

  useEffect(() => {
    if (!videoRef.current || !window.Holistic || !window.Camera) return;

    const holistic = new window.Holistic({
      locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });
    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults(results => {
      const raw = extractRawLeftHandLandmarks(results);
      if (!raw || !shouldSendToPython.current) return;

      const features = extractVectorFeatures3D_PythonEquivalent(raw);
      if (!features) return;

      setSequence(seq => {
        const next = [...seq, features];
        if (next.length === 10 && !collectingRef.current) {
          collectingRef.current = true;

          fetch('http://localhost:5001/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence: next })
          })
            .then(res => res.json())
            .then(data => {
              const pred = data.prediction || '';
              setPrediction(pred);
              setConfidence(data.confidence || 0);

              if (pred === targetLetterRef.current && lastMatchedLetter.current !== targetLetterRef.current) {
                setCorrectCount(prev => {
                  const newCount = prev + 1;
                  correctCountRef.current = newCount;
                  logDebugInfo(pred, newCount);
                  if (newCount >= 8) {
                    setResultMessage('🎉 게임 종료! 8개 정답!');
                    stopGame();
                  } else {
                    setResultMessage('✅ 정답입니다!');
                    const nextLetter = getRandomLetter();
                    setTimeout(() => {
                      setTargetLetter(nextLetter);
                      targetLetterRef.current = nextLetter;
                      lastMatchedLetter.current = '';
                      console.log("➡️ 다음 문제:", nextLetter);
                    }, 1000);
                  }
                  return newCount;
                });
                lastMatchedLetter.current = targetLetterRef.current;
              } else if (pred !== targetLetterRef.current) {
                setResultMessage('❌ 오답입니다. 다시 시도하세요.');
                logDebugInfo(pred, correctCount);
              }
            })
            .catch(err => console.error("❌ 예측 에러:", err))
            .finally(() => {
              collectingRef.current = false;
              setSequence([]);
            });
          return [];
        }
        return next.length > 10 ? next.slice(-10) : next;
      });
    });

    holisticRef.current = holistic;

    const camera = new window.Camera(videoRef.current, {
      onFrame: async () => {
        if (!videoRef.current || videoRef.current.readyState < 2 || !holisticRef.current) return;
        try {
          await holisticRef.current.send({ image: videoRef.current });
        } catch (err) {
          console.error("send 중 예외:", err);
        }
      },
      width: 640,
      height: 480
    });

    camera.start();
    cameraRef.current = camera;

    return () => {
      cameraRef.current?.stop();
      holisticRef.current?.close();
    };
  }, []);

  return (
    <div style={{ display: "flex", backgroundColor: "#fffff2", minHeight: "100vh" }}>
      <div className="sidebar">
        <button><Link to="/quiz">수어 맞추기</Link></button>
        <button><Link to="/text-to-sign" className="link">지문자 맞추기</Link></button>
        <button><Link to="/scoreboard">스코어</Link></button>
      </div>

      <div style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "flex-start",
        padding: "2rem",
        minHeight: "100vh",
        transition: "all 0.5s ease"
      }}>
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
            className="quiz-title"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            style={{ fontSize: "30px", margin: 0 }}
          >
            지문자 맞추기 게임 🎮
          </motion.h1>

          <motion.button
            onClick={isGameRunning ? stopGame : startGame}
            className="quiz-button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            style={{
              fontSize: "16px",
              padding: "10px 20px",
              cursor: "pointer",
              backgroundColor: isGameRunning ? "#ff4d4d" : "#4caf50",
              color: "#fff",
              border: "none",
              borderRadius: "6px"
            }}
          >
            {isGameRunning ? "게임 종료" : "게임 시작"}
          </motion.button>
        </motion.div>

        <motion.div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "40px",
            marginBottom: "30px"
          }}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <motion.div
            style={{
              fontSize: "54px",
              fontWeight: "bold",
              backgroundColor: "#e6f3d8",
              padding: "20px 40px",
              borderRadius: "12px",
              border: "2px solid #bdd8a7",
              textAlign: "center",
              height: "70px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              minWidth: "120px"
            }}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.8 }}
          >
            {targetLetter}
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <div style={{
              height: "12px", width: "360px", backgroundColor: "#eee",
              borderRadius: "6px", marginBottom: "10px"
            }}>
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
            <div style={{ fontWeight: "bold", fontSize: "18px" }}>
              ⏱️ 남은 시간: {timeLeft}초 | 🏆 점수: {correctCount}
            </div>
          </motion.div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          style={{
            width: "800px",
            height: "800px",
            backgroundColor: "#000",
            marginBottom: "40px",
            borderRadius: "12px",
            overflow: "hidden"
          }}
        >
          <video
            autoPlay
            playsInline
            muted
            ref={videoRef}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          style={{ textAlign: "center" }}
        >
          <div style={{ fontSize: "20px", fontWeight: "bold", marginBottom: "1rem" }}>
            {resultMessage}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
