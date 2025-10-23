/*
// LoginPage.jsx
import React, { useState } from 'react';
import axios from '../api/axios';
import { useNavigate } from 'react-router-dom';

const LoginPage = ({ setIsLoggedIn }) => {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      const res = await axios.post('/api/login', {
        userId,
        password
      }, {
        withCredentials: true
      });

      console.log('✅ 로그인 응답:', res.data);

      if (res.data === '로그인 성공') {
        setIsLoggedIn(true); // ✅ 로그인 상태 업데이트
        alert('로그인 성공');
        navigate(-1);
      } else {
        alert('아이디 또는 비밀번호가 잘못되었습니다.');
      }
    } catch (err) {
      console.error('❌ 로그인 실패:', err);
      alert('로그인 중 오류가 발생했습니다.');
    }
  };

  return (
    <div style={{ maxWidth: '400px', margin: '0 auto', padding: '40px' }}>
      <h2>로그인</h2>
      <form onSubmit={handleLogin}>
        <div style={{ marginBottom: '16px' }}>
          <label>아이디</label>
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            required
            style={{ width: '100%', padding: '8px' }}
          />
        </div>
        <div style={{ marginBottom: '16px' }}>
          <label>비밀번호</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{ width: '100%', padding: '8px' }}
          />
        </div>
        <button type="submit" style={{ padding: '10px 20px' }}>로그인</button>
      </form>
    </div>
  );
};

export default LoginPage;
*/
// LoginPage.jsx
import React, { useState } from 'react';
import axios from '../api/axios';
import { useNavigate } from 'react-router-dom';
import '../css/LoginPage.css'; // CSS 파일 임포트

const LoginPage = ({ setIsLoggedIn }) => {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  // const [rememberMe, setRememberMe] = useState(false); // 로그인 상태 유지 (필요시)
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('/api/login', { userId, password }, { withCredentials: true });
      if (res.data === '로그인 성공') {
        setIsLoggedIn(true);
        alert('로그인 성공');
        navigate(-1); // 이전 페이지로 이동
      } else {
        alert('아이디 또는 비밀번호가 잘못되었습니다.');
      }
    } catch (err) {
      console.error('❌ 로그인 실패:', err);
      alert('로그인 중 오류가 발생했습니다.');
    }
  };

  // 회원가입 페이지로 이동하는 함수 (예시)
  const goToSignUp = () => {
    navigate('/signup'); // 실제 회원가입 경로로 변경
  };

  // 아이디/비밀번호 찾기 페이지로 이동하는 함수 (예시)
  const goToFindCredentials = () => {
    navigate('/find-credentials'); // 실제 찾기 경로로 변경
  };


  return (
    // body에 배경색을 적용했으므로, 여기에는 별도 div가 필요 없을 수 있습니다.
    // 만약 body에 직접 적용하기 어렵다면, 이 div에 배경색과 정렬 스타일을 주세요.
    // <div className="login-page-wrapper"> 
      <div className="Login_main">

        <label className="Log_In">Log In</label>
        <label className='Login'>로그인</label>
        <form onSubmit={handleLogin}>
          <div className="L_form_box">
            <input
              type="text"
              placeholder="아이디"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              required
            />
            <input
              type="password"
              placeholder="비밀번호"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />

            {/* 로그인 상태 유지 (필요시 추가)
            <div className="form_remember_me">
              <input 
                type="checkbox" 
                id="rememberMe" 
                checked={rememberMe} 
                onChange={(e) => setRememberMe(e.target.checked)} 
              />
              <label htmlFor="rememberMe">로그인 상태 유지</label>
            </div>
            */}
            
          
            <button type="submit">로그인</button>
            
            <div className="form_links">
              <a href="/signup" onClick={goToSignUp}>회원가입</a> 
            <a href="/find-credentials" onClick={goToFindCredentials}>아이디/비밀번호 찾기</a>
            </div>


          </div>
        </form>
      </div>
    // </div>
  );
};

export default LoginPage;