import streamlit as st
import json


def load_media_from_urls(urls_json):
    try:
        media_data = json.loads(urls_json)
        return media_data
    except json.JSONDecodeError:
        st.error("잘못된 JSON 형식입니다.")
        return []

def show_gallery():
    st.title("🖼️ Bedrock Gallery")
    
    with st.sidebar.expander("**이미지 생성**", icon='📤', expanded=True):
        st.button("생성하기")
    
    with st.sidebar.expander("**갤러리 설정**", icon='⚙️', expanded=True):
        options = ["이미지 보기", "비디오 보기"]
        selection = st.segmented_control("미디어 필터", options, selection_mode="multi", 
                                       default=["이미지 보기", "비디오 보기"])
        cols_per_row = st.slider("열 수 설정", min_value=1, max_value=5, value=3)
    
    # Load and display media
    media_data = {
        "media": [
            {
                "type": "image",
                "url": "https://dje861stkudh8.cloudfront.net/c59ad840-ba53-4889-8829-df2a0c14d5fd.png",
                "title": "이미지 1",
                "metadata": {"hi": "hello"},
            }
        ]
    }
    
    _display_media_grid(media_data, selection, cols_per_row)

def _display_media_grid(media_data, selection, cols_per_row):
    if not media_data or 'media' not in media_data:
        st.warning("유효한 미디어 데이터가 없습니다.")
        return
    
    show_images = "이미지 보기" in selection
    show_videos = "비디오 보기" in selection
    media_files = [m for m in media_data['media']
                   if (m['type'] == 'image' and show_images) or
                      (m['type'] == 'video' and show_videos)]
    
    cols = st.columns(cols_per_row)
    for idx, media in enumerate(media_files):
        col_idx = idx % cols_per_row
        
        with cols[col_idx]:
            try:
                _display_media_item(media)
            except Exception as e:
                st.error(f"미디어 로드 중 오류 발생: {str(e)}")

def _display_media_item(media):
    container = st.container()
    if media['type'] == 'image':
        container.image(media['url'], caption=media['title'])
    elif media['type'] == 'video':
        container.video(media['url'], loop=True, autoplay=True)
    
    with container.expander("INFO", icon="🔎"):
        st.json(media['metadata'])