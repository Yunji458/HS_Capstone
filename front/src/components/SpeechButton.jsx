import React, { useRef } from "react";
import { Button } from "../css/Styles";

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

export default function SpeechButton({setText, setIsListening, isListening}) {
    const recognitionRef = useRef(null);
        if(!SpeechRecognition){
            alert("음성 인식을 지원하지 않음.");
            console.error("SpeechRecognition이 지원되지 않음.");
            return;
        }
        if(!recognitionRef.current){
            const recognition = new SpeechRecognition();
            recognition.lang = "ko-KR";
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;
            recognition.onresult = (event) => {
                const result = event.results[0][0].transcript;
                console.log("음성 인식 결과:", result);
                setText(result);
            };
            recognition.onerror = (event) => {
                console.error("음성 인식 오류:", event.error);
                alert("음성 인식 중 오류 발생: " + event.error);
                setIsListening(false);
            };

            recognition.onend = () => {
                setIsListening(false);
                
            };
            recognitionRef.current = recognition;
        }
    const startRecognition = () => {
        try {
            recognitionRef.current.start();
            setIsListening(true);
        } catch(error) {
            console.error("음성 인식 시작 오류:", error);
        }
    };
    return (
        <div>
        <button style={{
            ...Button,
            width: "120px",
            padding: "12px 20px",
            borderRadius: "25px",
            fontSize: "16px",
            fontWeight: "bold",
            backgroundColor: "#7e9e89",
            color: "#fff",
            border: "none",
            cursor: "pointer",
            transition: "background-color 0.3s"
        }} onClick={startRecognition}>
            음성 인식

        </button>
        <p>{isListening ? "음성 인식 중..." : " "}</p>
        </div>
    );
}