import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';

/* 추가 내용 */

import './index.css';
import Header from "./components/Header.jsx";
import Footer from "./components/Footer.jsx";
import reportWebVitials from './reportWebVitals.js';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <BrowserRouter>
      <App />
    <Footer attr={"footer__wrap"} />
  </BrowserRouter>
);

reportWebVitials();