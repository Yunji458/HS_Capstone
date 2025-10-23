import React, { useEffect, useRef, useState } from 'react';

export default function FingerspellCollectPage() {
  const videoRef = useRef(null);
  const collectingRef = useRef(false);
  const holisticRef = useRef(null);
  const cameraRef = useRef(null);
  const actionRef = useRef('');
  const [action, setAction] = useState('');
  const [timer, setTimer] = useState(0);
  const [framesCollected, setFramesCollected] = useState(0);
  const [sequence, setSequence] = useState([]);
  const [status, setStatus] = useState('');

  function extractKeypoints(results) {
    const flatten = (landmarks) => {
      if (!landmarks) return new Array(21 * 3).fill(0);
      return landmarks.flatMap(lm => [lm.x, lm.y, lm.z]);
    };
    const lh = flatten(results.leftHandLandmarks);
    const rh = flatten(results.rightHandLandmarks);
    return [...lh, ...rh]; // 총 126개
  }

  useEffect(() => {
    if (!videoRef.current || !window.Holistic || !window.Camera) return;

    if (!holisticRef.current) {
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

          if (next.length === 10) {
            console.log('▶ 전송 시작', { action: actionRef.current, sequence: next });

            fetch('http://localhost:5001/collect', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ action: actionRef.current, sequence: next }),
            })
              .then(res => res.json())
              .then(data => {
                console.log('✅ 응답', data);
                setStatus('✅ 수집 완료 및 증강됨!');
              })
              .catch(err => {
                console.error(err);
                setStatus('❌ 오류 발생');
              })
              .finally(() => {
                collectingRef.current = false;
                setSequence([]);
                setFramesCollected(0);
              });
          }

          return next;
        });
      });

      holisticRef.current = holistic;
    }

    if (!cameraRef.current) {
      const camera = new window.Camera(videoRef.current, {
        onFrame: async () => {
          await holisticRef.current.send({ image: videoRef.current });
        },
        width: 640,
        height: 480,
      });

      camera.start();
      cameraRef.current = camera;
    }

    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
    };
  }, []);

  const startCollect = () => {
    if (!action.trim()) return alert('라벨을 입력하세요.');
    actionRef.current = action.trim();
    setTimer(3);
    setStatus('');
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
    <div className="collect-container">
      <h1>지문자 좌표 수집</h1>
      <input
        type="text"
        placeholder="지문자 라벨"
        value={action}
        onChange={e => setAction(e.target.value)}
      />
      <button onClick={startCollect} disabled={collectingRef.current}>
        {collectingRef.current ? '수집 중...' : '수집 시작'}
      </button>
      {timer > 0 && <p>준비: {timer}초</p>}
      {collectingRef.current && <p>수집된 프레임: {framesCollected}/10</p>}
      {status && <p>{status}</p>}
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
