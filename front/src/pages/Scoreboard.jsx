import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../css/Scoreboard.css';

const ScoreItem = ({ rank, nickname, imageBase64, sum }) => (
  <li className="score-item">
    <span className="rank">{rank}위</span>
    {imageBase64 ? (
      <img src={imageBase64} alt={`${nickname} avatar`} className="avatar small-avatar" />
    ) : (
      <div className="avatar small-avatar placeholder-avatar">?</div>
    )}
    <span className="user-id">{nickname}</span>
    <span className="score">{sum.toLocaleString()}점</span>
  </li>
);

const PodiumItem = ({ rank, nickname, imageBase64, sum, position }) => (
  <div className={`podium-item podium-${position}`}>
    <div className="podium-rank-badge">{rank === 1 ? '👑' : ''} {rank}위</div>
    <img
      src={imageBase64 || 'https://via.placeholder.com/80/DDD/000?Text=NoImg'}
      alt={`${nickname} avatar`}
      className="avatar large-avatar"
    />
    <div className="podium-user-id">{nickname}</div>
    <div className="podium-score">{sum.toLocaleString()}점</div>
  </div>
);

const Scoreboard = () => {
  const [scores, setScores] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8080/api/users/top', { withCredentials: true })
      .then(res => setScores(res.data))
      .catch(err => console.error("Error loading scores:", err));
  }, []);

  const rank1 = scores[0];
  const rank2 = scores[1];
  const rank3 = scores[2];
  const others = scores.slice(3);

  if (scores.length === 0) {
    return <div className="scoreboard-container"><p>아직 기록이 없습니다.</p></div>;
  }

  return (
    <div className="scoreboard-container">
      <h2>🏆 명예의 전당 🏆</h2>
      <div className="podium-section">
        {rank2 && <PodiumItem {...rank2} rank={2} position="second" />}
        {rank1 && <PodiumItem {...rank1} rank={1} position="first" />}
        {rank3 && <PodiumItem {...rank3} rank={3} position="third" />}
      </div>
      <div className="others-list-section">
        <h3>TOP 10 (4위 이하)</h3>
        <ul>
          {others.map((user, index) => (
            <ScoreItem key={user.nickname} rank={index + 4} {...user} />
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Scoreboard;
