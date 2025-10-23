import React from "react";
import "../../css/MainPage.css";
import { Link, Element } from "react-scroll";
import Page1 from "./Page1";
import Page2 from "./Page2";
import Page3 from "./Page3";
import Page4 from "./Page4";
import { motion } from "framer-motion";

const MainPage = () => {
  const sectionVariants = {
    hidden: { opacity: 0, y: 100},
    visible: { opacity: 1, y: 0},
  }
  return (
    <div>
      <div className="mainbox">

        <div className="marquee">
          <div className="marquee-content big-bold">
            <span>SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL</span>
            <span>SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL</span>
          </div>
        </div>
        
        <div className="marquee">
          <div className="marquee-content reverse medium">
            <span>Sign For You LANGUAGE With Us 2025. HanSung University Sign For You LANGUAGE With Us 2025.</span>
            <span>Sign For You LANGUAGE With Us 2025. HanSung University Sign For You LANGUAGE With Us 2025.</span>
          </div>
        </div>

        <div className="marquee">
          <div className="marquee-content big-bold">
            <span>SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL</span>
            <span>SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL</span>
          </div>
        </div>

        <div className="marquee">
          <div className="marquee-content reverse medium">
            <span>Sign For You LANGUAGE With Us 2025. HanSung University Sign For You LANGUAGE With Us 2025.</span>
            <span>Sign For You LANGUAGE With Us 2025. HanSung University Sign For You LANGUAGE With Us 2025.</span>
          </div>
        </div>

        <div className="marquee">
          <div className="marquee-content big-bold">
            <span>SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL</span>
            <span>SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL SIGN:AL</span>
          </div>
        </div>

        <div>
          <br/><br/><br/>
          <span style={{fontSize:'26px', color:'#11574A'}}>↓ scroll</span>
        </div>
      </div>

    <div className="Mainbottom">
      <nav className="Mainpage_">
        <Link to="sectionOne" smooth={true} duration={1000} spy={true} activeClass="active">
            <span>SIGN:AL</span>
          </Link>
          <Link to="sectionTwo" smooth={true} duration={1000} spy={true} activeClass="active">
            <span>설명</span>
          </Link>
          <Link to="sectionThree" smooth={true} duration={1000} spy={true} activeClass="active">
            <span>개요</span>
          </Link>
          <Link to="sectionFour" smooth={true} duration={1000} spy={true} activeClass="active">
            <span>팀원 소개</span>
          </Link>
      </nav>

      {/* 페이지 1 */}
        <Element name="sectionOne"> {/* Element로 감싸고 name 속성 지정 */}
          <motion.section
            // id="section1" // Element의 name을 사용하므로 id는 선택사항
            variants={sectionVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: false, amount: 0.3 }}
            transition={{ duration: 1, ease: "easeOut" }}
            style={{ height: "100vh", backgroundColor: "#11574A", display: "flex", alignItems: "center", justifyContent: "center" }}
          >
            <Page1 />
          </motion.section>
        </Element>

        {/* 페이지 2 */}
        <Element name="sectionTwo"> {/* Element로 감싸고 name 속성 지정 */}
          <motion.section
            // id="section2"
            variants={sectionVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: false, amount: 0.3 }}
            transition={{ duration: 1, ease: "easeOut", delay: 0.1 }}
            style={{ height: "100vh", backgroundColor: "#11574A", display: "flex", alignItems: "center", justifyContent: "center" }}
          >
            <Page2 />
          </motion.section>
        </Element>

        {/* 페이지 3 */}
        <Element name="sectionThree"> {/* Element로 감싸고 name 속성 지정 */}
          <motion.section
            // id="section3"
            variants={sectionVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: false, amount: 0.3 }}
            transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
            style={{ height: "100vh", backgroundColor: "green", display: "flex", alignItems: "center", justifyContent: "center" }}
          >
            <Page3 />
          </motion.section>
        </Element>

        {/* 페이지 4 */}
        <Element name="sectionFour"> {/* Element로 감싸고 name 속성 지정 */}
          <motion.section
            // id="section4"
            variants={sectionVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: false, amount: 0.3 }}
            transition={{ duration: 1, ease: "easeOut", delay: 0.3 }}
            style={{ height: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}
          >
            <Page4 />
          </motion.section>
        </Element>
      </div>
    </div>
  );
};

export default MainPage;