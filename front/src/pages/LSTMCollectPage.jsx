import React, { useEffect, useRef, useState } from 'react';

export default function CollectPage() {
  const videoRef = useRef(null);
  const holisticRef = useRef(null);
  const collectingRef = useRef(false);
  const isSending = useRef(false);
  const actionRef = useRef('');

  const [action, setAction] = useState('');
  const [collecting, setCollecting] = useState(false);
  const [framesCollected, setFramesCollected] = useState(0);
  const [timer, setTimer] = useState(0);
  const [sequence, setSequence] = useState([]);
  const [training, setTraining] = useState(false);

  const HAND_SIZE = 21;

  const extractKeypoints = results => {
    const flatten = (landmarks, expectedCount) => {
      if (!landmarks) return Array(expectedCount * 3).fill(0);
      return [...Array(expectedCount).keys()].flatMap(i => {
        const lm = landmarks[i];
        return lm ? [lm.x, lm.y, lm.z] : [0, 0, 0];
      });
    };

    const nose = results.poseLandmarks ? [results.poseLandmarks[0]] : [];
    const noseFlattened = flatten(nose, 1);
    const leftHandFlattened = flatten(results.leftHandLandmarks, HAND_SIZE);
    const rightHandFlattened = flatten(results.rightHandLandmarks, HAND_SIZE);
    return [...noseFlattened, ...leftHandFlattened, ...rightHandFlattened];
  };

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
      if (!collectingRef.current || isSending.current) return;

      const keypoints = extractKeypoints(results);

      setSequence(seq => {
        if (seq.length >= 30) return seq;

        const next = [...seq, keypoints];
        setFramesCollected(next.length);

        if (next.length === 30) {
          collectingRef.current = false;
          setCollecting(false);
          isSending.current = true;

          console.log("▶ 전송 시작", { action: actionRef.current, sequence: next });

          fetch('http://localhost:8080/api/collect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: actionRef.current, sequence: next }),
          })
            .then(res => res.json().then(console.log))
            .catch(console.error)
            .finally(() => {
              isSending.current = false;
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

    holisticRef.current = holistic;
    camera.start();

    return () => {
      camera.stop();
    };
  }, []); // 최초 1회만 실행

  const startCollect = () => {
    if (!action.trim()) return alert('라벨을 입력하세요.');
    actionRef.current = action; // ✅ 현재 라벨값 보존

    setTimer(3);

    const countdown = setInterval(() => {
      setTimer(t => {
        if (t <= 1) {
          clearInterval(countdown);
          collectingRef.current = true;
          setCollecting(true);
          setSequence([]);
          setFramesCollected(0);
          return 0;
        }
        return t - 1;
      });
    }, 1000);
  };

  const startTraining = async () => {
    setTraining(true);
    try {
      const res = await fetch('http://localhost:8080/api/train', { method: 'POST' });
      const data = await res.text();
      alert(data);
    } catch (err) {
      console.error('학습 오류:', err);
      alert('학습 중 오류 발생');
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="collect-container">
      <h1>프레임 수집</h1>
      <input
        type="text"
        placeholder="동작 라벨"
        value={action}
        onChange={e => {
          setAction(e.target.value);
          actionRef.current = e.target.value; // ✅ 항상 최신 action 유지
        }}
      />
      <button onClick={startCollect} disabled={collecting}>
        {collecting ? '수집중...' : '수집 시작'}
      </button>
      <button onClick={startTraining} disabled={training}>
        {training ? '학습 중...' : '학습하기'}
      </button>
      {timer > 0 && <p>준비: {timer}초</p>}
      {collecting && <p>수집된 프레임: {framesCollected}/30</p>}
      <video
        ref={videoRef}
        className="output_video"
        style={{ width: 640, height: 480 }}
        autoPlay
        muted
      />
    </div>
  );
}
