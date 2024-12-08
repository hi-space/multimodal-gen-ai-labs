import streamlit as st
from datetime import datetime
from session import get_history, clear_history


if 'request_history' not in st.session_state:
    st.session_state.request_history = []

st.session_state.request_history.insert(0, {
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'request_type': "이미지 생성",
    'details': {
        "prompt": "user_prompt",
        "model": "selected_model",
        "parameters": {...}
    }
})

st.session_state.request_history.insert(0, {
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'request_type': "비디오 생성",
    'details': {
        "prompt": "user_prompt",
        "model": "selected_model",
        "parameters": {...}
    }
})


def show_history():
    st.title("📋 Request History")

    
    # Filtering options
    col1, col2 = st.columns([2, 1])
    with col1:
        history = get_history()
        filter_type = st.multiselect(
            "요청 유형 필터",
            options=list(set(item['request_type'] for item in history)) if history else [],
            default=list(set(item['request_type'] for item in history)) if history else []
        )
    
    with col2:
        if st.button("새로고침", type="secondary"):
            clear_history()
            st.rerun()
    
    _display_history_items(filter_type)

def _display_history_items(filter_type):
    history = get_history()
    
    if history:
        filtered_history = [item for item in history 
                          if item['request_type'] in filter_type]
        
        for item in filtered_history:
            with st.expander(
                f"**{item['request_type']}** - {item['timestamp']}", 
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
        st.markdown("**상세 정보:**")
    with col2:
        st.text(item['timestamp'])
        st.text(item['request_type'])
        st.json(item['details'])