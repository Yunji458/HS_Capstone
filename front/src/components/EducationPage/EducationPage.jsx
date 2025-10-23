import React, { useEffect, useRef, useState } from "react";
import { Swiper, SwiperSlide } from "swiper/react";
import "swiper/css";
import "../../css/Education.css";
import { Grid } from "swiper/modules";
import "swiper/css/grid";

export default function SwiperCard() {
  const swiperRef = useRef(null);
  const [swiperList, setSwiperList] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredTableData, setFilteredTableData] = useState([]);
  // topic state는 현재 직접 사용되지 않으므로 유지하거나 필요시 활용
  // const [topic, setTopic] = useState(null); 
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [selectedCardData, setSelectedCardData] = useState(null); // 선택된 카드 데이터 상태
  const [activeCategory, setActiveCategory] = useState('');

  const handleCategorySelect = (categoryName, event) => {
    if (event) event.stopPropagation(); // 이벤트 전파 중지
    setActiveCategory(categoryName);    // UI 업데이트를 위해 활성 카테고리 상태 변경
    handleTopic(categoryName);          // 기존 로직 실행
  };


  // 검색 처리
  const handleSearch = () => {
    if (searchQuery.trim() === "") {
      setFilteredTableData([]); // 검색어가 없으면 테이블 비움
      return;
    }
    fetch(`http://localhost:8080/api/education/search?name=${searchQuery}`)
      .then((res) => res.json())
      .then((data) => {
        setFilteredTableData(data);
        setSelectedCardData(null); // 검색 결과 표시 시, 선택된 카드 뷰 해제
      })
      .catch((error) => console.error("Search error:", error));
  };

  // 토픽 클릭 시 해당 토픽으로 데이터 필터링
  const handleTopic = (topic) => {
     if (topic === '전체') {
    fetch(`http://localhost:8080/api/education/all`) 
      .then((res) => res.json())
      .then((data) => {
        setFilteredTableData(data);
      })
      .catch((error) => console.error("Fetch all data error:", error));
  } else {
    fetch(`http://localhost:8080/api/education/category?category=${topic}`)
      .then((res) => res.json())
      .then((data) => {
        setFilteredTableData(data);
        setSelectedCardData(null); // 카테고리 선택 시, 선택된 카드 뷰 해제
      })
      .catch((error) => console.error("Category filter error:", error));
  };
}

  // Swiper 카드 클릭 시 (확대 표시용)
  const handleSwiperCardClick = (e, data) => {
    e.stopPropagation(); // 배경 클릭 이벤트 전파 방지
    setSelectedCardData(data);
    setFilteredTableData([]); // 테이블 필터링된 내용 초기화
  };

  // 배경 클릭 또는 닫기 버튼 클릭 시
  const handleCloseSelectedCard = (e) => {
    if (e) e.stopPropagation(); // 이벤트가 있다면 전파 방지
    setSelectedCardData(null);
  };

  // 전체 배경 클릭 처리
  const handleBackgroundClick = () => {
    if (selectedCardData) {
      setSelectedCardData(null); // 선택된 카드가 있으면 닫기
    } else {
      setFilteredTableData([]); // 없으면 테이블 비우기
    }
  };


  useEffect(() => {
    fetch("http://localhost:8080/api/education")
      .then((res) => res.json())
      .then((data) => {
        setSwiperList(data);
      })
      .catch((error) => console.error("Error fetching data:", error));
  }, []);

  return (
    <div className="Edu_Main" onClick={handleBackgroundClick}>
      {/* 검색창 */}
      <div className="Edu_Content_Wrap">
        <label>단어찾기</label>
        <div className="Edu_Content">
          <input
            className="searchbar"
            type="text"
            placeholder="검색어를 입력하시오"
            value={searchQuery}
            onClick={(e) => e.stopPropagation()} // 검색창 클릭 시 배경 클릭 방지
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()} // 엔터키로 검색
          />
          <button className="search-but" onClick={(e) => { e.stopPropagation(); handleSearch();}}>
            🔍
          </button>
        </div>
      </div>

      {/* 카테고리 테이블: selectedCardData가 없을 때만 표시 */}
      {!selectedCardData && (
        <div className="Edu_TableChart">
          
          <div className="category-picker">
            {['전체','학교', '병원', '일상', '집'].map((categoryName) => (
              <div
                key={categoryName}
                className={`category-button ${activeCategory === categoryName ? 'active' : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  handleCategorySelect(categoryName, e);
                }}
              >
                <span className="dot"></span>
                <span className="label">{categoryName}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 필터링된 테이블 출력: selectedCardData가 없고, filteredTableData가 있을 때만 */}
      {!selectedCardData && filteredTableData.length > 0 && (
        <div className="Edu_TableChart">
          <table className="chart filtered-results-table">
            <thead>
              <tr>
                <th>이름</th>
                <th>이미지 및 비디오</th>
              </tr>
            </thead>
            <tbody>
              {filteredTableData.map((data, idx) => (
                <tr key={idx} onClick={(e) => e.stopPropagation()}> {/* 테이블 행 클릭 시 배경 클릭 방지 */}
                  <td>{data.name}</td>
                  <td style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <img src={data.imageUrl} alt={data.name} style={{ width: "150px", display: "block", margin: "auto", marginBottom: "10px" }} />
                    <video src={data.videoUrl} style={{ width: "300px", display: "block", margin: "auto" }} controls />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* 선택된 카드 크게 보기 */}
      {selectedCardData && (
        <div className="selected-card-overlay" onClick={handleCloseSelectedCard}>
          <div className="expanded-card-container" onClick={(e) => e.stopPropagation()}>
            <button className="close-expanded-card" onClick={handleCloseSelectedCard}>
              X
            </button>
            <h2 className="expanded-card-name">{selectedCardData.name}</h2>
            <img src={selectedCardData.imageUrl} alt={selectedCardData.name} className="expanded-card-img" />
            <video src={selectedCardData.videoUrl} className="expanded-card-video" controls />
          </div>
        </div>
      )}

      {/* Swiper: selectedCardData가 없을 때만 표시 */}
      {!selectedCardData && swiperList && swiperList.length > 0 && (
        <div className="swiper-container">
          <button className="prev-btn" onClick={(e) => { e.stopPropagation(); swiperRef.current?.slidePrev(); }}>
            Prev
          </button>
          <Swiper
            onBeforeInit={(swiper) => {
              swiperRef.current = swiper;
            }}
            modules={[Grid]}
            slidesPerView={5}
            grid={{ rows: 2, fill: "row" }}
            spaceBetween={20}
            className="education-swiper"
          >
            {swiperList.map((data, idx) => (
              <SwiperSlide key={idx} onClick={(e) => handleSwiperCardClick(e, data)}>
                <div
                  className="square-card"
                  onMouseEnter={() => setHoveredIndex(idx)}
                  onMouseLeave={() => setHoveredIndex(null)}
                >
                  <div className={`card-inner ${hoveredIndex === idx ? 'is-flipped' : ''}`}>
                    <div className="card-front">
                      <p className="card-name">{data.name}</p>
                    </div>
                    <div className="card-back">
                      <img src={data.imageUrl} alt={data.name} className="card-img" />
                    </div>
                  </div>
                </div>
              </SwiperSlide>
            ))}
          </Swiper>
          <button className="next-btn" onClick={(e) => { e.stopPropagation(); swiperRef.current?.slideNext(); }}>
            Next
          </button>
        </div>
      )}
    </div>
  );
}