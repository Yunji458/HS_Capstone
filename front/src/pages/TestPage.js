import React, { useEffect, useRef, useState } from 'react';

export default function TestPage() {
  const videoRef = useRef(null);
  const collectingRef = useRef(false);
  const [timer, setTimer] = useState(0);
  const [framesCollected, setFramesCollected] = useState(0);
  const [sequence, setSequence] = useState([]);
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState(null);
  const [predictedText, setPredictedText] = useState('');

  const faceIndices = [1, 33, 263, 61, 291];
  const poseIndices = [11, 12];
  const handIndices = Array.from({ length: 21 }, (_, i) => i);

  function extractKeypoints(results) {
    const flatten = (landmarks, indices) => {
      if (!landmarks) return new Array(indices.length * 3).fill(0);
      return indices.flatMap(i => {
        const lm = landmarks[i];
        if (!lm) return [0, 0, 0];
        return [lm.x, lm.y, lm.z];
      });
    };

    const face = flatten(results.faceLandmarks, faceIndices);
    const lh = flatten(results.leftHandLandmarks, handIndices);
    const rh = flatten(results.rightHandLandmarks, handIndices);
    const pose = flatten(results.poseLandmarks, poseIndices);

    return [...face, ...lh, ...rh, ...pose]; // length === 147
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

          fetch('http://localhost:8080/api/predict', {
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
    <div className="test-container" style={{ display: 'flex', gap: '20px' }}>
      <div>
        <h1>수어 예측</h1>
        <button onClick={startPredict} disabled={collectingRef.current}>
          {collectingRef.current ? '예측 중...' : '예측 시작'}
        </button>
        {timer > 0 && <p>준비: {timer}초</p>}
        {collectingRef.current && <p>수집된 프레임: {framesCollected}/30</p>}
        {prediction && (
          <p>
            ✅ 예측 결과: <strong>{prediction}</strong> (
            <strong>{confidence}%</strong>)
          </p>
        )}
        <video
          ref={videoRef}
          className="output_video"
          style={{ width: 640, height: 480 }}
          autoPlay
          muted
        />
      </div>
      <div>
        <h2>출력된 문장</h2>
        <textarea
          value={predictedText}
          onChange={e => setPredictedText(e.target.value)}
          style={{ width: 300, height: 480, fontSize: '1.2rem', padding: '10px' }}
        />
      </div>
    </div>
  );
}
