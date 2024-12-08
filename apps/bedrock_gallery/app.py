import os
import sys
sys.path.append(os.path.abspath("../../"))

import streamlit as st
from config import config
from genai_kit.utils.images import encode_image_base64, base64_to_bytes
from genai_kit.aws.bedrock import BedrockModel

from styles import load_styles

from components.gallery import show_gallery
from components.image_generator import show_image_generator
from components.video_generator import show_video_generator
from components.history import show_history


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
    
    with gallery_tab:
        show_gallery()

    with image_generator_tab:
        show_image_generator()

    with video_generator_tab:
        show_video_generator()

    with history_tab:
        show_history()

if __name__ == "__main__":
    main()