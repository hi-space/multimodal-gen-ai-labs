import os
import sys
sys.path.append(os.path.abspath("../../"))

from apps.bedrock_gallery.types import MediaType
import streamlit as st
from config import config
from genai_kit.utils.images import encode_image_base64, base64_to_bytes
from genai_kit.aws.bedrock import BedrockModel

from styles import load_styles

from components.gallery import show_gallery
from components.image_generator import show_image_generator
from components.video_generator import show_video_generator
from components.history import show_history
from session import SessionManager


def get_media_items(session_manager: SessionManager, filter_type):
    history = session_manager.get_history()
    if history:
        media_items = [item for item in history 
                          if item['media_type'] in filter_type]
        
        media_items = sorted(
            media_items,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )
    return media_items


def main():
    st.set_page_config(page_title="Bedrock Media Gallery", layout="wide")
    load_styles()

    st.sidebar.title("Bedrock Media Gallery")
    st.sidebar.caption("Made by [hi-space](https://github.com/hi-space/multimodal-gen-ai-labs.git)")
    
    st.sidebar.info("""📚 **Requirements:** 
- Bedrock Model Access
- S3 Bucket
- CloudFront distribution (for S3)
- DynamoDB Table
""")
    
    # Create tabs
    image_generator_tab, video_generator_tab, gallery_tab, history_tab = st.tabs([
        "🎨 Generator", "🎥 Generator", "🖼️ Gallery", "📋 History"
    ])
    
    with st.sidebar.expander("**설정**", icon='⚙️', expanded=True):
        enum_values = [type.value for type in MediaType]
        filter_type = st.multiselect(
            "요청 유형 필터",
            options=enum_values,
            default=enum_values,
        )

        # Layout settings
        cols_per_row = st.slider("열 수 설정", min_value=1, max_value=5, value=3)
        
        # Detail view toggle
        show_details = st.checkbox("상세 정보 표시", value=False)

    session_manager = SessionManager()

    with image_generator_tab:
        show_image_generator(session_manager)

    with video_generator_tab:
        show_video_generator(session_manager)

    media_items = get_media_items(session_manager, filter_type)    
    with gallery_tab:
        show_gallery(media_items, cols_per_row, show_details)

    with history_tab:
        show_history(media_items, show_details)


if __name__ == "__main__":
    main()