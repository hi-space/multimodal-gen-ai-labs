import os
import sys
sys.path.append(os.path.abspath("../../"))

import streamlit as st
from typing import List
from genai_kit.aws.amazon_image import ImageParams, TitanImageSize, NovaImageSize
from genai_kit.aws.bedrock import BedrockModel
from genai_kit.utils.images import encode_image_base64, base64_to_bytes
from services.bedrock_service import (
    gen_english,
    gen_mm_image_prompt,
    gen_image,
    create_image_params
)
from session import add_to_history


def show_image_generator():
    st.title("🎨 Image Generator")
    _initialize_session_state()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        _show_prompt_section()
    
    with col2:
        _show_image_prompt_section()
    
    with col3:
        generate_clicked = _show_model_section()
    
    if generate_clicked:
        _show_generated_images_section()

def _initialize_session_state():
    """Initialize session state variables"""
    if 'image_prompt' not in st.session_state:
        st.session_state.image_prompt = ""
    if 'selected_colors' not in st.session_state:
        st.session_state.selected_colors = []
    if 'use_colors' not in st.session_state:
        st.session_state.use_colors = False
    if 'model_type' not in st.session_state:
        st.session_state.model_type = ""
    if 'generation_configs' not in st.session_state:
        st.session_state.generation_configs = None

def _show_prompt_section():
    """Display the prompt input section"""
    st.subheader("Generate a Prompt")
    prompt_type = st.selectbox(
        "Choose an option:",
        ["Basic Prompt", "Augmented Prompt"]
    )

    with st.expander("LLM Configuration"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9)
        top_k = st.slider("Top K", 0, 500, 250, 5)
    
    if prompt_type == "Basic Prompt":
        prompt_text = st.text_area("Enter your prompt:", height=150)
        if st.button("Generate Prompt", icon='📝', type="primary", use_container_width=True):
            with st.spinner("Generating prompt..."):
                st.session_state.image_prompt = gen_english(
                    request=prompt_text,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
    else:  # Augmented Prompt
        multimodal_keyword_text = st.text_area("Enter the keyword:", height=100)
        reference_image = st.file_uploader("Upload a reference image:", type=['png', 'jpg', 'jpeg'])
        
        if reference_image:
            st.image(reference_image, caption="Reference Image")
            
        if st.button("Generate Prompt", icon='📝', type="primary", use_container_width=True):
            with st.spinner("Generating prompt..."):
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

def _show_image_prompt_section():
    """Display the generated image prompt section"""
    st.subheader("Image Prompt")
    prompt_value = st.text_area(
        label="Edit or modify the prompt",
        value=st.session_state.image_prompt,
        height=400
    )

    if prompt_value != st.session_state.image_prompt:
        st.session_state.image_prompt = prompt_value

def _show_model_section():
    """Display the model configuration and generation section"""
    st.subheader("Select a Model")
    model_type = st.selectbox(
        "Choose a model:",
        [BedrockModel.TITAN_IMAGE.value, BedrockModel.NOVA_CANVAS.value]
    )

    st.session_state.model_type = model_type

    with st.expander("Image Configuration", expanded=True):
        configs = _get_model_configurations(model_type)
        st.session_state.generation_configs = configs
    
    return st.button("Generate Images", icon='🎨', type="primary", use_container_width=True)

def _get_model_configurations(model_type):
    """Get model configurations from user input"""
    num_images = st.slider("Number of Images", 1, 5, 1)
    cfg_scale = st.slider("CFG Scale", 1.0, 10.0, 8.0, 0.5)
    seed = st.number_input("Seed", 0, 2147483646, 0)
    
    if model_type == BedrockModel.TITAN_IMAGE.value:
        size_enum = TitanImageSize
    elif model_type == BedrockModel.NOVA_CANVAS.value:
        size_enum = NovaImageSize
    size_options = {f"{size.width} X {size.height}": size for size in size_enum}
    selected_size = st.selectbox("Image Size", options=list(size_options.keys()))
    
    use_colors = st.checkbox("Using color references")
    selected_colors = []
    
    if use_colors:
        selected_colors = _handle_color_selection()
        st.session_state.use_colors = True
        st.session_state.selected_colors = selected_colors
    else:
        st.session_state.use_colors = False
        st.session_state.selected_colors = []
    
    return {
        'num_images': num_images,
        'cfg_scale': cfg_scale,
        'seed': seed,
        'size': size_options[selected_size],
        'use_colors': use_colors,
        'selected_colors': selected_colors
    }

def _show_generated_images_section():
    """Display the generated images section"""
    st.divider()
    st.subheader("Generated Images")
    
    with st.status("Generating images...", expanded=True) as status:
        try:
            configs = st.session_state.generation_configs
            img_params = create_image_params(
                seed=configs['seed'],
                count=configs['num_images'],
                width=configs['size'].width,
                height=configs['size'].height,
                cfg=configs['cfg_scale']
            )

            configuration = img_params.get_configuration()
            
            if configs['use_colors']:
                body = img_params.color_guide(
                    text=st.session_state.image_prompt, 
                    colors=configs['selected_colors']
                )
                configuration['colorGuide'] = configs['selected_colors']
            else:
                body = img_params.text_to_image(
                    text=st.session_state.image_prompt
                )
            
            model_type = BedrockModel(st.session_state.model_type)
            imgs = gen_image(body=body, modelId=model_type)
            
            # Display prompt
            st.info(st.session_state.image_prompt)
            
            cols = st.columns(len(imgs))
            for idx, img in enumerate(imgs):
                with cols[idx]:
                    image_data = base64_to_bytes(img)
                    st.image(image_data, use_container_width=True)
            
            # Add to history
            add_to_history("이미지 생성", {
                "prompt": st.session_state.image_prompt,
                "model": "Titan Image v2",
                "request": configuration
            })
            
            status.update(label="Generation completed!", state="complete")
            
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            st.error(f"이미지 생성 중 오류가 발생했습니다: {str(e)}")
        finally:
            st.session_state.is_generating_image = False

def _handle_color_selection() -> List[str]:
    """Handle color selection UI and logic"""
    color_picker = st.color_picker("Pick a color")
    
    if color_picker and color_picker not in st.session_state.selected_colors:
        st.session_state.selected_colors.append(color_picker)
    
    selected_colors = st.multiselect(
        "Selected Colors",
        options=st.session_state.selected_colors,
        default=st.session_state.selected_colors,
        max_selections=10
    )
    
    if selected_colors:
        _display_color_preview(selected_colors)
    
    return selected_colors

def _display_color_preview(colors: List[str]):
    """Display color preview boxes"""
    color_html = "<div style='display: flex; flex-wrap: wrap;'>"
    for color in colors:
        color_html += (
            f"<div style='width: 24px; height: 24px; background-color: {color}; "
            f"margin-left: 5px; margin-bottom: 8px; border-radius: 5px;'></div>"
        )
    color_html += "</div>"
    st.markdown(color_html, unsafe_allow_html=True)