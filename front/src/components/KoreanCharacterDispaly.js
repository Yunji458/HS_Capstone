import React, { useEffect, useState, useRef } from 'react';
import { Client } from '@stomp/stompjs';
import Hangul from 'hangul-js';

function KoreanCharacterDisplay() {
  const [inputLetters, setInputLetters] = useState([]);
  const [finalWord, setFinalWord] = useState('');
  const videoRef = useRef(null);

  // 웹캠 스트림 설정
  useEffect(() => {
    async function setupWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('웹캠 접근 에러:', err);
      }
    }
    setupWebcam();
  }, []);

  // STOMP 메시지 수신
  useEffect(() => {
    const socketUrl = 'ws://localhost:8080/ws';
    const stompClient = new Client({
      webSocketFactory: () => new WebSocket(socketUrl),
      reconnectDelay: 5000,
      debug: (str) => console.log(str),
    });

    stompClient.onConnect = () => {
      console.log("웹소켓 연결 성공");
      stompClient.subscribe('/topic/characters', (message) => {
        if (message.body) {
          setInputLetters(prev => [...prev, message.body]);
        }
      });
    };

    stompClient.activate();

    return () => {
      stompClient.deactivate();
    };
  }, []);

  const handleLetterRemove = (index) => {
    setInputLetters(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    const composed = Hangul.assemble(inputLetters);
    setFinalWord(composed);
    setInputLetters([]);
  };

  const handleReadAloud = () => {
    if (!finalWord) return;
    const utterance = new SpeechSynthesisUtterance(finalWord);
    utterance.lang = 'ko-KR';
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div style={{
      backgroundColor: "#f5fde9",
      textAlign: "center",
      minHeight: "100vh"
    }}>
      <h2 style={{ fontSize: "30px", marginBottom: "30px", color: "#3c4b3f" }}>
        수어 인식 및 단어 조합
      </h2>

      <div style={{
        display: "flex",
        alignItems: "stretch",
        justifyContent: "center",
        flexWrap: "wrap",
        gap: "40px"
      }}>
        {/* 비디오 창 */}
        <div style={{
          width: "640px",
          height: "480px",
          borderRadius: "16px",
          marginTop: "30px",
          backgroundColor: "#000",
          boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
          overflow: "hidden"
        }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              borderRadius: "16px"
            }}
          />
        </div>

        {/* 입력된 자모 박스 */}
        <div style={{
          width: "320px",
          minHeight: "300px",
          padding: "30px",
          borderRadius: "16px",
          backgroundColor: "#f7fff6",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)",
          display: "flex",
          marginTop: "30px",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          gap: "20px"
        }}>
          <h3 style={{ marginBottom: "10px", color: "#3a5e4c" }}>입력된 자모</h3>
          <div style={{
            minHeight: "50px",
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "center",
            gap: "8px"
          }}>
            {inputLetters.map((letter, index) => (
              <span
                key={index}
                onClick={() => handleLetterRemove(index)}
                style={{
                  padding: '6px 12px',
                  background: '#e0f1e8',
                  border: '1px solid #a7c8b5',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '16px'
                }}
              >
                {letter}
              </span>
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '16px' }}>
            <button
              onClick={handleSubmit}
              style={{
                padding: '12px 20px',
                fontSize: '16px',
                backgroundColor: '#7fa68c',
                color: '#fff',
                border: 'none',
                borderRadius: '25px',
                cursor: 'pointer',
                width: '120px'
              }}
            >
              제출
            </button>
            <button
              onClick={handleReadAloud}
              style={{
                padding: '12px 20px',
                fontSize: '16px',
                backgroundColor: '#a8bfa9',
                color: '#fff',
                border: 'none',
                borderRadius: '25px',
                cursor: 'pointer',
                width: '120px'
              }}
            >
              음성읽기
            </button>
          </div>
        </div>
      </div>

      {/* 최종 단어 출력 */}
      {finalWord && (
        <div style={{ marginTop: '40px' }}>
          <h2 style={{ fontSize: '24px', color: '#3c5c44' }}>최종 단어</h2>
          <p style={{
            fontSize: '36px',
            fontWeight: 'bold',
            color: '#2e4635',
            marginTop: '10px',
          }}>
            {finalWord}
          </p>
        </div>
      )}
    </div>
  );
}

export default KoreanCharacterDisplay;
