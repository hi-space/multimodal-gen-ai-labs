import streamlit as st
import json
from session import SessionManager, MediaType
from utils import format_datetime

session_manager = SessionManager()


def show_history():
    st.title("📋 Request History")

    # Filtering options
    col1, col2 = st.columns([2, 1])
    with col1:
        enum_values = [type.value for type in MediaType]
        filter_type = st.multiselect(
            "요청 유형 필터",
            options=enum_values,
            default=enum_values,
        )
    
    with col2:
        if st.button("새로고침", type="secondary"):
            session_manager.clear_history()
            st.rerun()
    
    _display_history_items(filter_type)

def _display_history_items(filter_type):
    history = session_manager.get_history(include_media=True)

    if history:
        filtered_history = [item for item in history 
                          if item['media_type'] in filter_type]
        
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
        st.markdown("**프롬프트:**")
        st.markdown("**상태:**")       
        st.markdown("**상세 정보:**")
    
    with col2:
        st.text(format_datetime(item['created_at'], seconds=True))
        st.text(item['media_type'])
        st.text(item.get('prompt', ''))
        st.text(item.get('status', 'UNKNOWN'))
        
        url = item.get('url', '')
        if url:
            st.image(url)
                
        st.json(json.loads(json.dumps(item['details'], default=float)))
