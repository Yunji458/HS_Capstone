import React from 'react';
import {Link} from 'react-router-dom'
import '../css/footer.css'; 


const Footer = (props) => {
  return (
    <footer className="footer" role="contentinfo" >
      <div className="footer__inner container">
        <div className="footer__text">
          <h5>초신성</h5>
          <p>
            2025_Capstone 
           
          </p>
          
        </div>
        <div className="footer__menu">
          <div>
            <h4>팀원</h4>
            <span>황하성</span>
            <span>김대환</span>
            <span>김윤지</span>
            <span>정다빈</span>
            <span>장세미</span>
            
          </div>
        
          
        </div>
        <div className="footer__right">
          Copyright 2025. All Rights Reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;