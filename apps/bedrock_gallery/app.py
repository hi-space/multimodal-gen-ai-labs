import os
import sys
sys.path.append(os.path.abspath("../../"))

import streamlit as st
import requests
import json
from datetime import datetime
from PIL import Image
from io import BytesIO

from config import config
from generator import gen_english, gen_mm_image_prompt, gen_image
from genai_kit.aws.amazon_image import (
    BedrockAmazonImage, ImageParams, TitanImageSize, NanoImageSize
)
from genai_kit.utils.images import encode_image_base64, base64_to_bytes
from genai_kit.aws.bedrock import BedrockModel


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

def show_video_generator():
    pass

def show_image_generator():
    st.title("🎨 Image Generator")
    image_prompt = ""

    if 'image_prompt' not in st.session_state:
        st.session_state.image_prompt = ""
    if 'selected_colors' not in st.session_state:
        st.session_state.selected_colors = []
    if 'use_colors' not in st.session_state:
        st.session_state.use_colors = False
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Prompt")
        prompt_type = st.selectbox(
            "Choose an option:",
            ["Basic Prompt", "Augmented Prompt"]
        )

        with st.expander("LLM Configuration"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            top_p = st.slider("Top P", 0.0, 1.0, 0.9)
            top_k = st.slider("Top K", 0, 500, 250, 5)
        
        if prompt_type == "Basic Prompt":
            prompt_text = st.text_area("Enter your prompt:")
        elif prompt_type == "Augmented Prompt":
            multimodal_keyword_text = st.text_area("Enter the keyword:")
            reference_image = st.file_uploader("Upload a reference image:")
            if reference_image:
                st.image(reference_image, caption="Uploaded Image")
            
        if st.button("Generate Prompt", type="primary", use_container_width=True):
            st.session_state.image_prompt = ""
            if prompt_type == "Basic Prompt":
                st.session_state.image_prompt = gen_english(
                    request=prompt_text,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
            elif prompt_type == "Augmented Prompt":
                image = None
                if reference_image:
                    image = encode_image_base64(reference_image)
                st.session_state.image_prompt = gen_mm_image_prompt(
                    keyword=multimodal_keyword_text,
                    image=image,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
    
    with col2:
        st.subheader("2. Image Prompt")
        if 'image_prompt' not in st.session_state:
            st.session_state.image_prompt = ""

        st.text_area(
            label="prompt",
            value=st.session_state.image_prompt,
            height=200
        )
            
    with col3:
        st.subheader("3. Model")
        prompt_type = st.selectbox(
            "Choose a model:",
            ["Titan Image v2", "Nano Canvas"]
        )

        st.subheader("Configurations")
        with st.expander("Configuration", expanded=True):
            num_images = st.slider("Number of Images", 1, 5, 1)
            cfg_scale = st.slider("CFG Scale", 1.0, 10.0, 8.0, 0.5)
            seed = st.number_input("Seed", 0, 2147483646, 0)
            size_options = {f"{size.value[0]} X {size.value[1]}": size for size in TitanImageSize}
            selected_size = st.selectbox("Image Size", options=list(size_options.keys()))
            
            use_colors = st.checkbox("Using color references")
            if use_colors:
                color_picker = st.color_picker("Pick a color")
                if 'selected_colors' not in st.session_state:
                    st.session_state.selected_colors = []
                if color_picker and color_picker not in st.session_state.selected_colors:
                    st.session_state.selected_colors.append(color_picker)
                
                st.session_state.selected_colors = st.multiselect(
                    "Selected Colors",
                    options=st.session_state.selected_colors,
                    default=st.session_state.selected_colors,
                    max_selections=10
                )
                
                if st.session_state.selected_colors:
                    color_html = "<div style='display: flex; flex-wrap: wrap;'>"
                    for color in st.session_state.selected_colors:
                        color_html += f"<div style='width: 24px; height: 24px; background-color: {color}; margin-left: 5px; margin-bottom: 8px; border-radius: 5px;'></div>"
                    color_html += "</div>"
                    st.markdown(color_html, unsafe_allow_html=True)
        
        if st.button("Generate Images", type="primary", use_container_width=True):
            generate_images(
                st.session_state.image_prompt,
                num_images,
                cfg_scale,
                seed,
                size_options[selected_size]
            )


def generate_images(image_prompt, num_images, cfg_scale, seed, selected_size):
    st.divider()
    st.subheader("Image Generation")
    with st.status("Generating...", expanded=True):
        img_params = ImageParams(seed=seed)
        img_params.set_configuration(
            count=num_images,
            width=selected_size.width,
            height=selected_size.height,
            cfg=cfg_scale
        )

        cfg = img_params.get_configuration()

        if st.session_state.use_colors:
            body = img_params.color_guide(text=image_prompt, colors=st.session_state.selected_colors)
            cfg['colorGuide'] = st.session_state.selected_colors
        else:
            body = img_params.text_to_image(text=image_prompt)

        imgs = gen_image(body=body, modelId=BedrockModel.TITAN_IMAGE)
        
        st.info(image_prompt)
        cols = st.columns(len(imgs))
        for idx, img in enumerate(imgs):
            with cols[idx]:
                image_data = base64_to_bytes(img)
                # image_data = BytesIO(base64.b64decode(img))

                st.image(image_data)
                
                # with st.spinner("Upload..."):
                #     upload_image(
                #         image=image_data,
                #         prompt=image_prompt,
                #         cfg=cfg,
                #         tags=tags
                #     )

    
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
    st.set_page_config(page_title="My App", layout="wide")
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

    st.sidebar.title("Amazon Bedrock Gallery")
    st.sidebar.caption("Made by [hi-space](https://github.com/hi-space/multimodal-gen-ai-labs.git)")
    
    st.sidebar.info("""📚 **Requirements:** 
- Bedrock Model Access
- S3 Bucket
- CloudFront distribution (for S3)
- DynamoDB Table
""")
    
    # 탭 생성
    image_generator_tab, gallery_tab, video_generator_tab, history_tab = st.tabs([
        "🖼️ Gallery", "🎨 Generator", "🎥 Generator", "📋 History"
    ])

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

    with image_generator_tab:
        show_image_generator()

    with video_generator_tab:
        show_video_generator()

if __name__ == "__main__":
    main()