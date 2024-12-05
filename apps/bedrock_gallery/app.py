import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import json
from datetime import datetime


st.markdown("""
    <style>
        /* 이미지 스타일링 */
        .stImage {
            border-radius: 8px !important;
            transition: transform 0.2s !important;
            cursor: pointer !important;
        }
        .stImage:hover {
            transform: scale(1.03) !important;
        }
        
        /* 이미지 캡션 스타일링 */
        .stImage img {
            border-radius: 8px !important;
        }
            
        video {
            border-radius: 8px !important;
        }
        
        /* 히스토리 카드 스타일링 */
        div[data-testid="stExpander"] {
            background-color: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
    </style>
""", unsafe_allow_html=True)

def load_media_from_urls(urls_json):
    try:
        media_data = json.loads(urls_json)
        return media_data
    except json.JSONDecodeError:
        st.error("잘못된 JSON 형식입니다.")
        return []

def show_gallery():
    st.title("🖼️ Bedrock Gallery")
    
    # expander 안에 설정 옵션들을 배치
    with st.sidebar.expander("**갤러리 설정**", icon='⚙️', expanded=True):
        # 필터 옵션
        options = ["이미지 보기", "비디오 보기"]
        selection = st.segmented_control("미디어 필터", options, selection_mode="multi", default=["이미지 보기", "비디오 보기"])
        
        # 열 수 설정
        cols_per_row = st.slider("열 수 설정", min_value=1, max_value=5, value=3)
    
    # 미디어 데이터 로드
    media_data = {
        "media": [
            {
                "type": "image",
                "url": "https://dje861stkudh8.cloudfront.net/c59ad840-ba53-4889-8829-df2a0c14d5fd.png",
                "title": "이미지 1",
                "metadata": {
                    "hi": "hello"
                },
            }
        ]
    }
    
    if not media_data or 'media' not in media_data:
        st.warning("유효한 미디어 데이터가 없습니다.")
        return
    
    # 필터링
    show_images = "이미지 보기" in selection
    show_videos = "비디오 보기" in selection
    media_files = [m for m in media_data['media']
                   if (m['type'] == 'image' and show_images) or
                      (m['type'] == 'video' and show_videos)]
    
    # 갤러리 그리드
    cols = st.columns(cols_per_row)
    for idx, media in enumerate(media_files):
        col_idx = idx % cols_per_row
        
        with cols[col_idx]:
            try:
                if media['type'] == 'image':
                    # 이미지 클릭 이벤트를 위한 컨테이너
                    container = st.container()
                    container.image(media['url'], caption=media['title'])
                    with container.expander("INFO",  icon="🔎"):
                        st.json(media['metadata'])
                    
                elif media['type'] == 'video':
                    # 비디오 클릭 이벤트를 위한 컨테이너
                    container = st.container()
                    container.video(media['url'], loop=True, autoplay=True)
                    with container.expander("INFO",  icon="🔎"):
                        st.json(media['metadata'])
                   
            except Exception as e:
                st.error(f"미디어 로드 중 오류 발생: {str(e)}")

def show_history():
    st.title("📋 Request History")

    # 세션 스테이트에 히스토리가 없으면 초기화
    if 'request_history' not in st.session_state:
        st.session_state.request_history = []
    
    # 히스토리 필터링 옵션
    col1, col2 = st.columns([2, 1])
    with col1:
        filter_type = st.multiselect(
            "요청 유형 필터",
            options=list(set(item['request_type'] for item in st.session_state.request_history)) if st.session_state.request_history else [],
            default=list(set(item['request_type'] for item in st.session_state.request_history)) if st.session_state.request_history else []
        )
    with col2:
        clear_history = st.button("새로고침", type="secondary")
        if clear_history:
            st.session_state.request_history = []
            st.rerun()
    
    # 히스토리 테이블 표시
    if st.session_state.request_history:
        filtered_history = [item for item in st.session_state.request_history 
                          if item['request_type'] in filter_type]
        
        for item in filtered_history:
            with st.expander(f"**{item['request_type']}** - {item['timestamp']}", expanded=False):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown("**요청 시간:**")
                    st.markdown("**요청 유형:**")
                    st.markdown("**상세 정보:**")
                with col2:
                    st.text(item['timestamp'])
                    st.text(item['request_type'])
                    st.json(item['details'])
                
    else:
        st.info("아직 요청 기록이 없습니다.")

def add_to_history(request_type, details):
    if 'request_history' not in st.session_state:
        st.session_state.request_history = []
    
    history_item = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'request_type': request_type,
        'details': details
    }
    st.session_state.request_history.insert(0, history_item)


def main():
    st.sidebar.title("Amazon Bedrock Gallery")
    st.sidebar.caption("Made by [hi-space](https://github.com/hi-space/multimodal-gen-ai-labs.git)")
    st.sidebar.info("""📚 **Note:** Please ensure you have both **Bedrock Model Access** and **S3 permissions** enabled.""")
    
    # 탭 생성
    gallery_tab, history_tab = st.tabs(["Gallery", "History"])

    add_to_history("이미지 생성", {
        "prompt": "user_prompt",
        "model": "selected_model",
        "parameters": {...}
    })

    add_to_history("이미지 생성", {
        "prompt": "user_prompt",
        "model": "selected_model",
        "parameters": {...}
    })
    
    with history_tab:
        show_history()
        
    with gallery_tab:
        show_gallery()

if __name__ == "__main__":
    main()