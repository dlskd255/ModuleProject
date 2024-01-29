import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import webbrowser
import json
import pandas as pd
from configparser import ConfigParser

# ConfigParser 객체 생성 및 config.toml 파일 읽기
config = ConfigParser()
config.read("config.toml")

# 읽어온 설정값 사용
server_port = config.get("server", "port", fallback=8501)

class CustomVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def process(self, frame):
        self.frame_count += 1
        # 이미지 좌우 반전
        flipped_frame = cv2.flip(frame.data, 1)
        # 밝기 증가
        brightness_factor = 1.2
        brightened_frame = cv2.convertScaleAbs(flipped_frame, alpha=brightness_factor, beta=0)
        return brightened_frame
    
# Streamlit의 SessionState 모듈을 사용하여 상태 유지
class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

#Data Pull and Functions
st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
    """,
    unsafe_allow_html=True)

@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
# 제목을 가운데 정렬하는 함수
def centered_title(title_text):
    return f"<h1 style='text-align:center;'>{title_text}</h1>"

#Options Menu
with st.sidebar:
    # 스타일을 적용할 클래스를 지정
    st.markdown(
        """
        <style>
        .sidebar-content {
            background-color: #f0f0f0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 옵션 메뉴 생성
    selected = option_menu('메뉴', ["메인 페이지", '걸어서 세계속으로', '주제2'], 
                          icons=['play-btn', 'search', 'kanban'], menu_icon='intersect', default_index=0) # info-circle
    lottie = load_lottiefile("similo3.json")
    st_lottie(lottie, key='loc')

# 메인 페이지
if selected == "메인 페이지":
    # Header
    st.markdown(centered_title('Main Page'), unsafe_allow_html=True)

    st.divider()

    # Use Cases
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header('Use Cases')
            st.markdown(
                """
                - _Remote work got you thinking about relocation?_
                - _Looking for a new vacation spot?_
                - _Conducting market research for product expansion?_
                - _Just here to play and learn?_
                """
            )
        with col2:
            lottie2 = load_lottiefile("place2.json")
            st_lottie(lottie2, key='place', height=300, width=300)

    st.divider()


# Search Page
if selected == "걸어서 세계속으로":
    st.markdown(centered_title('걸어서 세계속으로'), unsafe_allow_html=True)
    st.divider()        

    # 설명서 버튼
    if st.button('설명서 보기🔍'):
        # 설명서 HTML 파일 경로
        documentation_path = 'C:/Users/blucom005/Downloads/정리폴더/24년도/프로젝트 문서/manual.html'

        # 새로운 브라우저 창에서 HTML 파일 열기
        webbrowser.open('file://' + documentation_path, new=2)

    # streamlit_webrtc를 사용하여 웹캠 표시
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=CustomVideoProcessor,  # CustomVideoProcessor를 사용하여 좌우 반전 및 밝기 조절
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        ),
    )

# About Page
if selected == '주제2':
    st.markdown(centered_title('주제2'), unsafe_allow_html=True)
    st.divider()
    st.write('설명')

# 터미널 명령어 : python -m streamlit run mainPage_v3.py