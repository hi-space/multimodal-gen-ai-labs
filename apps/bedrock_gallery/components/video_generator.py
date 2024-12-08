import streamlit as st
from session import add_to_history


def show_video_generator():
    st.title("🎥 Video Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Video Configuration")
        with st.form("video_config_form"):
            prompt = st.text_area("Video Prompt", 
                                placeholder="Enter your video generation prompt...")
            
            st.subheader("Settings")
            duration = st.slider("Duration (seconds)", 1, 30, 5)
            fps = st.slider("Frames per second", 15, 60, 30)
            
            with st.expander("Advanced Settings"):
                quality = st.select_slider("Quality", 
                                         options=["Low", "Medium", "High"], 
                                         value="Medium")
                style = st.selectbox("Style", 
                                   ["Realistic", "Cartoon", "Abstract", "Cinematic"])
            
            generate_button = st.form_submit_button("Generate Video")
            
            if generate_button:
                with st.spinner("Generating video..."):
                    # Add video generation logic here
                    add_to_history("비디오 생성", {
                        "prompt": prompt,
                        "duration": duration,
                        "fps": fps,
                        "quality": quality,
                        "style": style
                    })
    
    with col2:
        st.subheader("Preview")
        st.info("Generated video will appear here")
        # Add video preview logic here