import React, { useEffect, useState } from 'react';
import axios from '../api/axios';

const Mypage = () => {
  const [userInfo, setUserInfo] = useState({
    id: '',
    userId: '',
    nickname: '',
    studentId: ''
  });

  useEffect(() => {
    axios.get('/api/userinfo', { withCredentials: true })
      .then(res => {
        setUserInfo(res.data);
      })
      .catch(err => {
        console.error("사용자 정보 불러오기 실패:", err);
      });
  }, []);

  return (
    <div style={{ maxWidth: '600px', margin: '0 auto', padding: '40px' }}>
      <h2>마이페이지</h2>
      <div style={{ marginTop: '20px', fontSize: '18px' }}>
        <p><strong>로그인 ID:</strong> {userInfo.userId}</p>
        <p><strong>닉네임:</strong> {userInfo.nickname}</p>
        <p><strong>학번:</strong> {userInfo.studentId}</p>
      </div>
    </div>
  );
};

export default Mypage;
