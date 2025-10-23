import React from 'react';
import { href, Link, useLocation, useNavigate } from 'react-router-dom';
import axios from '../api/axios';
import '../css/Header.css';
import setting from "../images/setting.png";
import sign from '../images/Logo.png';

const Header = ({ isLoggedIn, setIsLoggedIn, attr }) => {
  const location = useLocation();
  const currentPath = location.pathname;
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await axios.post('/api/logout');
      alert('로그아웃 되었습니다.');
      setIsLoggedIn(false);
    } catch (err) {
      console.error('로그아웃 실패:', err);
      alert('로그아웃 실패');
    }
  };

  return (
    <header id="header" className={attr} role="heading" aria-level="1">
      <div className="header__inner container">
        <div className="header_logo">
          <Link to="/"><img src={sign} height="100px" width="100px" alt="logo" /></Link>
        </div>
        <div className="header__nav" role="navigation">
          <ul className="main_ul">
            <li className={currentPath === '/transport' ? 'active' : ''}>
              <Link to="/transport">번역하기</Link>
            </li>
            <li className={currentPath === '/quiz' ? 'active' : ''}>
              <Link to="/quiz">게임하기</Link>
            </li>
            <li className={currentPath === '/education' ? 'active' : ''}>
              <Link to="/education">단어찾기</Link>
            </li>
          </ul>
        </div>
        <div>
          {isLoggedIn ? (
            <>
              <button
                onClick={() => navigate('/mypage')}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer'
                }}
                title='마이페이지'
                >
                  마이페이지
              </button>
              
              <button
                onClick={handleLogout}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  marginRight: '10px'
                }}
                title='로그아웃'
                >
                  로그아웃
                </button>
            </>
          ) : (
            <button
              onClick={() => navigate('/login')}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer'
              }}
              title="로그인"
            >
              로그인
            </button>
          )}
          <button 
          onClick={() => navigate('/join')}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer'
              }}>
              회원가입
            </button>
        </div>

      </div>
    </header>
  );
};

export default Header;
