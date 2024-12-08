import streamlit as st
from datetime import datetime

def add_to_history(request_type, details):
    if 'request_history' not in st.session_state:
        st.session_state.request_history = []
    
    history_item = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'request_type': request_type,
        'details': details
    }
    st.session_state.request_history.insert(0, history_item)

def get_history():
    return st.session_state.get('request_history', [])

def clear_history():
    st.session_state.request_history = []