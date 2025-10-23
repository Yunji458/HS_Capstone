import React from "react";
import "../../css/Page3.css";

const Page3 = () => {
  return (
    <div className="page3-container">
      <div className="first-box">
        <p className="title-text">What is Sign Language?<br /></p>
        <p className="desc-text">Sign language is a visual language that uses hand shapes, facial expressions, and body movements to convey meaning..
          <br /> It is a fully developed language with its own grammar, vocabulary, and cultural significance, widely used by the Deaf and hard of hearing communities around the world. <br />
        </p>
       
      </div>
      <div className="second-box">
        <p className="title-text">About Our Sign Language Translator<br /></p>
        <p className="desc-text">Our sign language translator helps bridge the communication gap between sign language users and non-signers.
          <br/>By recognizing sign gestures through advanced AI technology, our system translates them into text or spoken language in real time, supporting smoother and more inclusive communication.<br/>
        </p>
      </div>

      <div className="third-box">
        <p className="title-text">Why Sign Language Matters<br /></p>
        <p className="desc-text">Sign language is more than just gestures—it is a vital means of expression and identity for Deaf communities.
          <br/>Promoting sign language not only enhances accessibility but also fosters respect, understanding, and inclusion in society.<br/>
        </p>
      </div>

    </div>
  );
};

export default Page3;
