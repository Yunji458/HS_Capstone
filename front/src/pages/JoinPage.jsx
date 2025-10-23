import React, { useState,useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import '../css/JoinPage.css';

const JoinPage = () => {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();
  const imgRef = useRef();
  const [previewImg, setPreviewImg] = useState('/profile.png');

// 이미지 업로드 input의 onChange
const saveImgFile = () => {
	const file = imgRef.current.files[0];
	const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
        setPreviewImg(reader.result);
   	};
};

  return (
    <div className="Join_main">
      <div className="Join_text">
      <label className='Join'>Join</label><br></br>
      <label className='Join_k'>회원가입</label>
      </div>
      <form>
        <img
        src={previewImg}
         style={{ width: '100px', height: '100px', objectFit: 'cover', borderRadius: '50%' }}/>
        <label className="signup-profileImg-label" htmlFor="profileImg">프로필 이미지 추가</label>
        <input
        className="signup-profileImg-input"
        type="file"
        accept="image/*"
        id="profileImg"
        ref={imgRef}
        onChange={saveImgFile}
         
        />
    </form>
      <form>
        <div className="form_box">
          <input
            type="text"
            placeholder="학번"
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
          <input
            type="password_pw"
            placeholder="비밀번호 확인"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <input
            type="name"
            placeholder="이름"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        <label>닉네임*</label>
        <input
            type="Nickname"
            placeholder="이름을 입력하세요."
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

        <button type="submit" style={{ padding: '10px 20px' }}>회원가입</button>
      </div>
      </form>
    </div>
  );
};

export default JoinPage;
