import React, { useEffect, useRef, useState } from 'react';
import Hangul from 'hangul-js';

function FingerspellDisplay() {
  const videoRef = useRef(null);
  const holisticRef = useRef(null);
  const cameraRef = useRef(null);
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [sequence, setSequence] = useState([]);
  const collectingRef = useRef(false);

  const [collectedLetters, setCollectedLetters] = useState([]);
  const [finalWord, setFinalWord] = useState('');

  function extractRawLeftHandLandmarks(results) {
    if (!results.leftHandLandmarks || results.leftHandLandmarks.length !== 21) {
      return null;
    }
    const landmarksNested = results.leftHandLandmarks.map(lm => [lm.x, lm.y, lm.z]);
    return landmarksNested.flat();
  }

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
        const raw_landmarks_flat = extractRawLeftHandLandmarks(results);
        if (!raw_landmarks_flat) return;

        const features = extractVectorFeatures3D_PythonEquivalent(raw_landmarks_flat);
        if (!features) return;

        setSequence(seq => {
          const next = [...seq, features];
          if (next.length === 10 && !collectingRef.current) {
            collectingRef.current = true;
            fetch('http://localhost:5001/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ sequence: next }),
            })
              .then(res => res.json())
              .then(data => {
                setPrediction(data.prediction || '');
                setConfidence(data.confidence || 0);
              })
              .catch(err => console.error('❌ 오류', err))
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
    }

    if (!cameraRef.current) {
      const camera = new window.Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current && holisticRef.current) {
            await holisticRef.current.send({ image: videoRef.current });
          }
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

  const handleAddLetter = () => {
    if (prediction) {
      setCollectedLetters(prev => [...prev, prediction]);
      setPrediction('');
    }
  };

  const handleMergeWord = () => {
    const word = Hangul.assemble(collectedLetters.join(''));
    setFinalWord(word);
  };

  return (
    <div style={{ backgroundColor: "#f5fde9", textAlign: "center", minHeight: "100vh" }}>
      <h2 style={{ fontSize: "30px", marginBottom: "30px", color: "#3c4b3f" }}>
        지문자 수어 예측
      </h2>

      <div style={{
        display: "flex",
        justifyContent: "center",
        flexWrap: "wrap",
        gap: "40px"
      }}>
        {/* 비디오 박스 */}
        <div style={{
          width: "640px",
          height: "550px",
          padding: "20px",
          borderRadius: "16px",
          backgroundColor: "#000",
          boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center"
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
              borderRadius: "12px"
            }}
          />
        </div>

        {/* 결과 출력 */}
        <div style={{
          width: "640px",
          height: "550px",
          padding: "30px",
          borderRadius: "16px",
          backgroundColor: "#f7fff6",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)",
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-start",
          alignItems: "center",
          gap: "20px"
        }}>
          <h3 style={{ fontSize: "24px", color: "#3a5e4c" }}>예측 결과</h3>
          <p style={{ fontSize: "36px", color: "#1a3e2a", fontWeight: "bold" }}>
            {prediction}
          </p>
          <p style={{ fontSize: "20px", color: "#3a5e4c" }}>
            신뢰도: {(confidence * 100).toFixed(2)}%
          </p>

          <button onClick={handleAddLetter} style={{
            padding: "10px 20px",
            fontSize: "16px",
            backgroundColor: "#7fa68c",
            color: "#fff",
            border: "none",
            borderRadius: "25px",
            cursor: "pointer"
          }}>
            자모 추가
          </button>

          <textarea
            value={collectedLetters.join(' ')}
            placeholder="인식된 자모가 여기에 쌓입니다"
            style={{
              width: "100%",
              height: "80px",
              fontSize: "18px",
              padding: "12px",
              borderRadius: "12px",
              border: "1px solid #a7c8b5",
              backgroundColor: "#ffffff",
              resize: "none"
            }}
          />

          <button onClick={handleMergeWord} style={{
            padding: "10px 30px",
            fontSize: "16px",
            backgroundColor: "#5d947c",
            color: "#fff",
            border: "none",
            borderRadius: "25px",
            cursor: "pointer"
          }}>
            단어 합치기
          </button>

          {finalWord && (
            <div>
              <h4 style={{ marginTop: "20px", color: "#3c5c44" }}>최종 단어</h4>
              <p style={{ fontSize: "28px", fontWeight: "bold", color: "#2e4635" }}>{finalWord}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default FingerspellDisplay;
