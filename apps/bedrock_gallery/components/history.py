import streamlit as st
import json
from apps.bedrock_gallery.session import SessionManager, MediaType
from apps.bedrock_gallery.utils import format_datetime


def show_history(session_manager: SessionManager):
    st.title("📋 Request History")

    with st.sidebar.expander("**히스토리 설정**", icon='⚙️', expanded=False):
        enum_values = [type.value for type in MediaType]
        filter_type = st.multiselect(
            "요청 유형 필터",
            options=enum_values,
            default=enum_values,
        )
        
    history = session_manager.get_history()

    if history:
        filtered_history = [item for item in history 
                          if item['media_type'] in filter_type]
        
        filtered_history = sorted(
            filtered_history,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )
        
        for item in filtered_history:
            with st.expander(
                f"**{item['media_type']}** - {format_datetime(item['created_at'])}",
                expanded=False
            ):
                _display_history_item(item)
    else:
        st.info("아직 요청 기록이 없습니다.")


def _display_history_item(item):
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**요청 시간:**")
        st.markdown("**요청 유형:**")
        st.markdown("**모델 타입:**")
        st.markdown("**프롬프트:**")
        st.markdown("**상세 정보:**")
    
    with col2:
        st.text(format_datetime(item['created_at'], seconds=True))
        st.text(item['media_type'])
        st.text(item['model_type'])
        st.code(item.get('prompt', ''), wrap_lines=True, language='txt')
        
        thumbnail = item.get('thumbnail', None)
        if thumbnail:
            st.image(thumbnail)
                
        st.json(json.loads(json.dumps(item['details'], default=float)))
