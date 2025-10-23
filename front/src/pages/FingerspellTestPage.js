import React, { useEffect, useRef, useState } from 'react';

export default function FingerspellTestPage() {
  const videoRef = useRef(null);
  const holisticRef = useRef(null);
  const cameraRef = useRef(null);
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [sequence, setSequence] = useState([]);
  const collectingRef = useRef(false);

  function extractRawLeftHandLandmarks(results) {
    if (!results.leftHandLandmarks || results.leftHandLandmarks.length !== 21) {
      return null;
    }
    const landmarksNested = results.leftHandLandmarks.map(lm => [lm.x, lm.y, lm.z]);
    return landmarksNested.flat();
  }

  function extractVectorFeatures3D_PythonEquivalent(landmarks_flat) {
    if (!landmarks_flat || landmarks_flat.length !== 21 * 3) {
        return null;
    }

    const getJoint = (index) => {
        const i = index * 3;
        return [landmarks_flat[i], landmarks_flat[i+1], landmarks_flat[i+2]];
    };

    const subtractVectors = (v1, v2) => [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]];

    const vectorNorm = (v) => Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

    const normalizeVector = (v) => {
        const norm = vectorNorm(v);
        if (norm < 1e-6) return [0, 0, 0];
        return [v[0]/norm, v[1]/norm, v[2]/norm];
    };

    const dotProduct = (v1, v2) => v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];

    const toDegrees = (radians) => radians * (180 / Math.PI);

    // 1. 정규화된 뼈대 벡터 (v)
    const v_source_indices1 = [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19];
    const v_source_indices2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
    const v_normalized_vectors = [];

    for (let i = 0; i < v_source_indices1.length; i++) {
        const joint1 = getJoint(v_source_indices1[i]);
        const joint2 = getJoint(v_source_indices2[i]);
        const vec = subtractVectors(joint1, joint2);
        const normalized_vec = normalizeVector(vec);
        v_normalized_vectors.push(...normalized_vec);
    }

    const getNormalizedBoneVector = (index_in_v_list) => {
        const i = index_in_v_list * 3;
        return [v_normalized_vectors[i], v_normalized_vectors[i+1], v_normalized_vectors[i+2]];
    }

    const angle_idx1 = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18];
    const angle_idx2 = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19];
    const angles = [];

    for (let i = 0; i < angle_idx1.length; i++) {
        const vec_i = getNormalizedBoneVector(angle_idx1[i]);
        const vec_j = getNormalizedBoneVector(angle_idx2[i]);
        let dot = dotProduct(vec_i, vec_j);
        dot = Math.max(-1.0, Math.min(1.0, dot));
        const angle_rad = Math.acos(dot);
        angles.push(toDegrees(angle_rad));
    }

    const base_indices = [5, 9, 13, 17];
    const inter_angles = [];
    const wrist_joint = getJoint(0);

    for (let i = 0; i < base_indices.length - 1; i++) {
        const mcp1_joint = getJoint(base_indices[i]); 
        const mcp2_joint = getJoint(base_indices[i+1]); 

        let vi = subtractVectors(mcp1_joint, mcp2_joint);
        vi = normalizeVector(vi);

        let vj = subtractVectors(wrist_joint, mcp1_joint);
        vj = normalizeVector(vj);

        let dot = dotProduct(vi, vj);
        dot = Math.max(-1.0, Math.min(1.0, dot)); 
        const angle_rad = Math.acos(dot);
        inter_angles.push(toDegrees(angle_rad));
    }

    return [...v_normalized_vectors, ...angles, ...inter_angles];
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

        if (!raw_landmarks_flat) {
          return;
        }
        
        const features = extractVectorFeatures3D_PythonEquivalent(raw_landmarks_flat);

        if (!features) {
            return;
        }

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
          return next.length > 10 ? next.slice(next.length - 10) : next;
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

  return (
    <div className="test-container">
      <h1>지문자 수어 예측</h1>
      <p>예측: {prediction} ({(confidence * 100).toFixed(2)}%)</p>
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
