import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import os
import webbrowser
import json
import pandas as pd
from configparser import ConfigParser
from PIL import Image

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.imgur.com/9BNbcc7.jpeg");
             background-attachment: fixed;
             background-size: cover
             
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

image_witw = "./walkintoworld.webp"

original_image = Image.open(image_witw)
resized_image = original_image.resize((original_image.width // 2, original_image.height // 2))

# ConfigParser 객체 생성 및 config.toml 파일 읽기
class CustomConfigParser(ConfigParser):
    def __init__(self):
        super().__init__()

    def get_config(self, section, key, default=None):
        try:
            return self.get(section, key)
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            return default

config = CustomConfigParser()
config.read("config.toml")

# 읽어온 설정값 사용
server_port = config.get("server", "port", fallback=8501)

# 앨범을 보여주는 함수
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            images.append(img_path)
    return images

def show_album():
    st.subheader("앨범")
    image_folder_path = "picture"
    images = load_images_from_folder(image_folder_path)

    if not images:
        st.warning("앨범에 이미지가 없습니다.")
    else:
        # 2열로 이미지 표시
        col1, col2 = st.columns(2)
        for i, image_path in enumerate(images):
            image = Image.open(image_path)
            if i % 2 == 0:
                col1.image(image, caption=os.path.basename(image_path), use_column_width=True)
            else:
                col2.image(image, caption=os.path.basename(image_path), use_column_width=True)

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
    selected = option_menu('메뉴', ["메인 페이지", '걸어서 세계속으로'], 
                          icons=['play-btn', 'search'], menu_icon='intersect', default_index=0) # info-circle
    lottie = load_lottiefile("similo3.json")
    st_lottie(lottie, key='loc')

# Apply custom font to the header
    st.markdown(
        """
        <style>
        .main-page-header {
            font-size: 80px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .sub-page-header {
            font-family: 'Caveat', cursive;
            font-size: 50px;
            text-align: center;
            
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .korean-text {
            font-family: 'MaruBuri-Regular', sans-serif;
            font-size: 50px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Main Page
if selected == "메인 페이지":
    # Header
    header_html = centered_title('Main Page')
    st.markdown("<p class='main-page-header'>수요일은 칼퇴</p>", unsafe_allow_html=True)

    st.divider()

    # Use Cases
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header('팀원 목록')
            st.markdown(
                """
                - _이승환_
                - _박세웅_
                - _이재범_
                - _최재권_
                - _황지의_
                """
            )
        with col2:
            lottie2 = load_lottiefile("place2.json")
            st_lottie(lottie2, key='place', height=300, width=300)

    st.divider()

# Search Page
if selected == "걸어서 세계속으로":
    st.image(image_witw, use_column_width=True)
    st.divider()        

    # 설명서 버튼
    if st.button('설명서 보기🔍'):
        # 설명서 HTML 파일 경로
        documentation_path = 'C:/Users/blucom005/Downloads/정리폴더/24년도/프로젝트 문서/manual.html'

        # 새로운 브라우저 창에서 HTML 파일 열기
        webbrowser.open('file://' + documentation_path, new=2)
    st.divider() 

    if st.button('앨범 보기'):
        show_album()

# 터미널 명령어 : python -m streamlit run mainPage_v3.py