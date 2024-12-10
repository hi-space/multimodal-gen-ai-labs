import streamlit as st
from datetime import datetime
from typing import Dict, Any, BinaryIO, Optional
from genai_kit.aws.bedrock import BedrockModel
from config import config
from services.storage_service import StorageService
from apps.bedrock_gallery.services.task_manager import AsyncTaskManager
from apps.bedrock_gallery.utils import extract_key_from_uri
from apps.bedrock_gallery.types import MediaType


class SessionManager:
    def __init__(self):
        self.storage_service = StorageService(
            bucket_name=config.S3_BUCKET,
            cloudfront_domain=config.CF_DOMAIN
        )

    def add_to_history(
        self,
        media_type: MediaType,
        prompt: str,
        model_type: BedrockModel, 
        details: Optional[Dict[str, Any]] = None,
        media_file: Optional[BinaryIO] = None,
        ref_image: Optional[str] = None,
    ):
        if 'request_history' not in st.session_state:
            st.session_state.request_history = []
        
        storage_metadata = None
        try:
            storage_metadata = self.storage_service.upload_media(
                media_type=media_type,
                model_type=model_type,
                prompt=prompt,
                details=details,
                media_file=media_file,
                ref_image=ref_image,
            )
        except Exception as e:
            st.error(f"Failed to upload media: {str(e)}")
            return None

        st.session_state.request_history.insert(0, storage_metadata)
        return storage_metadata
    
    def get_history(self, media_type: str = None):
        return self.storage_service.get_media_list(
            media_type=media_type,
        )

    def clear_history(self):
        st.session_state.request_history = []
