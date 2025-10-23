import React, { useEffect, useRef, useState } from 'react';

function LSTMDisplay() {
  const videoRef = useRef(null);
  const collectingRef = useRef(false);
  const [timer, setTimer] = useState(0);
  const [framesCollected, setFramesCollected] = useState(0);
  const [sequence, setSequence] = useState([]);
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState(null);
  const [predictedText, setPredictedText] = useState('');

  const HAND_KEYPOINTS = 21;
  const NUM_COORDS = 3;

  function extractKeypoints(results) {
    const extract = (landmarks, count) => {
      if (!landmarks || landmarks.length < count) return Array(count * NUM_COORDS).fill(0);
      return landmarks.slice(0, count).flatMap(lm => [lm.x, lm.y, lm.z]);
    };

    const poseNose = results.poseLandmarks && results.poseLandmarks[0]
      ? [results.poseLandmarks[0].x, results.poseLandmarks[0].y, results.poseLandmarks[0].z]
      : [0, 0, 0];

    const leftHand = extract(results.leftHandLandmarks, HAND_KEYPOINTS);
    const rightHand = extract(results.rightHandLandmarks, HAND_KEYPOINTS);

    return [...poseNose, ...leftHand, ...rightHand]; // 총 129개
  }

  useEffect(() => {
    if (!videoRef.current || !window.Holistic || !window.Camera) return;

    const holistic = new window.Holistic({
      locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
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
      if (!collectingRef.current) return;
      const keypoints = extractKeypoints(results);

      setSequence(seq => {
        const next = [...seq, keypoints];
        setFramesCollected(next.length);

        if (next.length === 30) {
          console.log("▶ 예측 요청", next);

          fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence: next }),
          })
            .then(res => res.json())
            .then(data => {
              if (data && data.result && data.confidence !== undefined) {
                setPrediction(data.result);
                setConfidence((data.confidence * 100).toFixed(2));
                setPredictedText(prev => prev + data.result);
              } else {
                console.error("Invalid response format:", data);
              }
            })
            .catch(console.error)
            .finally(() => {
              collectingRef.current = false;
              setSequence([]);
              setFramesCollected(0);
            });
        }

        return next;
      });
    });

    const camera = new window.Camera(videoRef.current, {
      onFrame: async () => {
        await holistic.send({ image: videoRef.current });
      },
      width: 640,
      height: 480,
    });

    camera.start();

    return () => {
      camera.stop();
    };
  }, []);

  const startPredict = () => {
    setTimer(3);
    setPrediction('');
    setConfidence(null);

    const countdown = setInterval(() => {
      setTimer(t => {
        if (t <= 1) {
          clearInterval(countdown);
          collectingRef.current = true;
          return 0;
        }
        return t - 1;
      });
    }, 1000);
  };

  return (
    <div style={{ backgroundColor: "#f5fde9", textAlign: "center", minHeight: "100vh" }}>
      <h2 style={{ fontSize: "30px", marginBottom: "30px", color: "#3c4b3f" }}>수어 예측</h2>

      <div style={{
        display: "flex", alignItems: "stretch", justifyContent: "center", flexWrap: "wrap", gap: "40px"
      }}>
        <div style={{
          width: "640px", height: "550px", padding: "20px", borderRadius: "16px",
          backgroundColor: "#000", boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
          display: "flex", alignItems: "center", justifyContent: "center"
        }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: "100%", height: "100%", objectFit: "cover", borderRadius: "16px"
            }}
          />
        </div>

        <div style={{
          width: "640px", height: "550px", padding: "30px", borderRadius: "16px",
          backgroundColor: "#f7fff6", boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)",
          display: "flex", flexDirection: "column", justifyContent: "center",
          alignItems: "center", gap: "20px"
        }}>
          <div style={{ width: "100%" }}>
            <h3 style={{ marginBottom: "10px", color: "#3a5e4c", textAlign: "center" }}>출력된 문장</h3>
            <textarea
              value={predictedText}
              onChange={e => setPredictedText(e.target.value)}
              style={{
                width: "90%", height: "320px", fontSize: "1.2rem",
                padding: "12px", borderRadius: "12px", border: "1px solid #a7c8b5",
                resize: "none", backgroundColor: "#ffffff"
              }}
            />
          </div>

          <button
            onClick={startPredict}
            disabled={collectingRef.current || timer > 0}
            style={{
              marginTop: "20px", padding: "12px 20px", fontSize: "16px",
              backgroundColor: "#7fa68c", color: "#fff", border: "none",
              borderRadius: "25px", cursor: "pointer", width: "100%", height: "50px"
            }}
          >
            {timer > 0
              ? `준비 중... (${timer}초)`
              : collectingRef.current
                ? `예측 중... (${framesCollected}/30)`
                : "예측 시작"}
          </button>
        </div>
      </div>

      {prediction && (
        <div style={{ marginTop: "30px", color: "#2f4f3e" }}>
          ✅ 예측 결과: <strong style={{ color: "#1a3e2a" }}>{prediction}</strong> (
          <strong>{confidence}%</strong>)
        </div>
      )}
    </div>
  );
}

export default LSTMDisplay;
