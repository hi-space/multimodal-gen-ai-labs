import streamlit as st
import json
from typing import List, Dict, Any
from session import SessionManager, MediaType
from utils import format_datetime


def show_gallery(session_manager: SessionManager):
    st.title("🖼️ GenAI Gallery")
    
    with st.sidebar.expander("**갤러리 설정**", icon='⚙️', expanded=False):
        # Media type filter
        media_types = [type.value for type in MediaType]
        selected_types = st.multiselect(
            "미디어 타입",
            options=media_types,
            default=media_types,
        )
        
        # Layout settings
        cols_per_row = st.slider("열 수 설정", min_value=1, max_value=5, value=3)
        
        # Detail view toggle
        show_details = st.checkbox("상세 정보 표시", value=False)
        
    media_items = session_manager.get_history()
    if media_items:
        filtered_media_items = [item for item in media_items 
                            if item['media_type'] in selected_types]
        
        filtered_media_items = sorted(
            filtered_media_items,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )

        _display_media_grid(filtered_media_items, cols_per_row, show_details)
    else:
        st.info("표시할 미디어가 없습니다.")


def _display_media_grid(media_items: List[Dict[str, Any]], cols_per_row: int, show_details: bool):
    cols = st.columns(cols_per_row)
    
    for idx, item in enumerate(media_items):
        col_idx = idx % cols_per_row
        
        with cols[col_idx]:
            _display_media_item(item, show_details)


def _display_media_item(item: Dict[str, Any], show_details: bool):
    container = st.container()
    
    media_type = item.get('media_type', '')
    url = item.get('url', '')
    
    if media_type == MediaType.IMAGE.value:
        container.image(url)
    elif media_type == MediaType.VIDEO.value:
        container.video(url)
    
    container.caption(f"_{item.get('prompt', '')}_")
    
    if show_details:
        with container.expander(format_datetime(item.get('created_at', ''), seconds=False), expanded=False):
            st.code(item.get('prompt', ''), wrap_lines=True, language='txt')
            
            st.markdown("**모델:**")
            st.text(item.get('model_type', ''))
            if details := item.get('details'):
                st.markdown("**상세 정보:**")
                st.json(json.loads(json.dumps(details, default=float)), expanded=False)