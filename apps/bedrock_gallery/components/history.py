import streamlit as st
import json
from typing import List
from apps.bedrock_gallery.session import SessionManager, MediaType
from apps.bedrock_gallery.utils import format_datetime


def show_history(media_items: List[dict] = [], show_details: bool = False):
    st.title("📋 Request History")

    for item in media_items:
        icon = _get_emoji(item['media_type'])
        with st.expander(
            f"**{item['media_type']}** - {format_datetime(item['created_at'])}",
            expanded=show_details,
            icon=icon,
        ):
            _display_history_item(item)

    else:
        st.info("아직 요청 기록이 없습니다.")

def _get_emoji(media_type: str):
    if media_type == MediaType.IMAGE.value:
        return '🎨'
    elif media_type == MediaType.VIDEO.value:
        return '🎥'
    return '❓'

def _display_history_item(item):
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**ID:**")
        st.markdown("**요청 시간:**")
        st.markdown("**요청 유형:**")
        st.markdown("**모델 타입:**")
        st.markdown("**프롬프트:**")
    
    with col2:
        media_type = item['media_type']
        st.text(item['id'])
        st.text(format_datetime(item['created_at'], seconds=True))
        st.text(media_type)
        st.text(item['model_type'])
        st.code(item.get('prompt', ''), wrap_lines=True, language='txt')
        
        url = item.get('url', '')
        if url and media_type == MediaType.IMAGE.value:
            st.image(url)
        elif url and media_type == MediaType.VIDEO.value:
            st.video(url)
                
        st.json(json.loads(json.dumps(item['details'], default=float)))
