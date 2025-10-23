/*
import React, { useEffect, useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import axios from './api/axios';

// 추가된 Header
import Header from './components/Header'; // 경로 확인 필수

// 페이지 컴포넌트
import Main from './components/MainPage/MainPage';
import Education2 from './components/EducationPage/EducationPage';
import Quiz from './pages/Quiz';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import KoreanCharacterDisplay from './components/KoreanCharacterDispaly';
import ToSign from './pages/ToSign';
import LSTMDisplay from './pages/LSTMDisplay';
import FingerspellDisplayStop from './pages/FingerspellDisplayStop';
import './App.css'
import Mypage from './pages/Mypage';
import FingerspellQuiz from './pages/fingerspellQuiz';

function App() {
  const [mode, setMode] = useState('signToText');
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    axios.get('/api/session', { withCredentials: true })
      .then(res => {
        setIsLoggedIn(res.data !== '세션 없음');
      })
      .catch(err => {
        console.error("세션 확인 실패:", err);
      });
  }, []);

  return (
    <div className='App'>
      // ✅ 여기에 Header 렌더링
      <Header isLoggedIn={isLoggedIn} setIsLoggedIn={setIsLoggedIn} />

      <Routes>
        <Route path='/' element={<Main />} />
        <Route path='/education' element={<Education2 />} />
        <Route path='/quiz' element={<Quiz />} />
        <Route path='/login' element={<LoginPage setIsLoggedIn={setIsLoggedIn} />} />
        <Route path='/register' element={<RegisterPage />} />
        <Route path='/mypage' element={<Mypage />} />

        <Route path='fingerquiz' element={<FingerspellQuiz />} />

        <Route
          path='/transport'
          element={
            <div className='translate-page'>
              <aside className='sidebar'>
                <button onClick={() => setMode('example')}>예시 페이지</button>
                <button onClick={() => setMode('signToTextLSTM')}>수어를 단어로 번역 (수어)</button>
                <button onClick={() => setMode('signToTextFingerspell')}>수어를 단어로 번역 (지문자)</button>
                <button onClick={() => setMode('textToSign')}>단어를 수어로 번역</button>
              </aside>
              <section className='content'>
                {mode === 'example' && <KoreanCharacterDisplay />}
                {mode === 'signToTextLSTM' && <LSTMDisplay />}
                {mode === 'signToTextFingerspell' && <FingerspellDisplayStop />}
                {mode === 'textToSign' && <ToSign />}
              </section>
            </div>
          }
        />
      </Routes>

    </div>
  );
}

export default App;
*/

import React, { useEffect, useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import axios from './api/axios';

// 추가된 Header
import Header from './components/Header'; // 경로 확인 필수

// 페이지 컴포넌트
import Main from './components/MainPage/MainPage';
import Education2 from './components/EducationPage/EducationPage';
import Quiz from './pages/Quiz';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import KoreanCharacterDisplay from './components/KoreanCharacterDispaly';
import ToSign from './pages/ToSign';
import LSTMDisplay from './pages/LSTMDisplay';
import FingerspellDisplayStop from './pages/FingerspellDisplayStop';
import './App.css'
import Mypage from './pages/Mypage';
import Join from './pages/JoinPage';
import Scoreboard from './pages/Scoreboard';
import TextToSignQuiz from './pages/TextToSign';

import CollectPage from './pages/CollectPage';
import LSTMCollectPage from './pages/LSTMCollectPage';

function App() {
  const [mode, setMode] = useState('signToText');
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    axios.get('/api/session', { withCredentials: true })
      .then(res => {
        setIsLoggedIn(res.data !== '세션 없음');
      })
      .catch(err => {
        console.error("세션 확인 실패:", err);
      });
  }, []);

  return (
    <div className='App'>
      {/* ✅ 여기에 Header 렌더링 */}
      <Header isLoggedIn={isLoggedIn} setIsLoggedIn={setIsLoggedIn} />

      <Routes>
        <Route path='/' element={<Main />} />
        <Route path='/education' element={<Education2 />} />
        <Route path='/quiz' element={<Quiz />} />
        <Route path='/login' element={<LoginPage setIsLoggedIn={setIsLoggedIn} />} />
        <Route path='/join' element={<Join />} />
        <Route path='/register' element={<RegisterPage />} />
        <Route path='/mypage' element={<Mypage />} />
        <Route path='/text-to-sign' element={<TextToSignQuiz />} />
        <Route path='/scoreboard' element={<Scoreboard />} />

        <Route path='/collect' element={<CollectPage />} />

        <Route
          path='/transport'
          element={
            <div className='translate-page'>
              <aside className='sidebar'>
                <button onClick={() => setMode('signToText')}>수어를 단어로 번역 (수어)</button>
                <button onClick={() => setMode('signToTextFingerspell')}>수어를 단어로 번역 (지문자)</button>
                <button onClick={() => setMode('textToSign')}>단어를 수어로 번역</button>
                <button onClick={() => setMode('custom')}>커스텀수어 만들기</button>
              </aside>
              <section className='content'>
                {mode === 'signToText' && <LSTMDisplay />}
                {mode === 'signToTextFingerspell' && <FingerspellDisplayStop />}
                {mode === 'textToSign' && <ToSign />}
                {mode === 'custom' && <LSTMCollectPage />}
              </section>
            </div>
          }
        />
      </Routes>
    </div>
  );
}

export default App;