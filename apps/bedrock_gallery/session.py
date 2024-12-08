import streamlit as st
from datetime import datetime
from enum import Enum
from typing import Dict, Any, BinaryIO, Optional
from genai_kit.aws.bedrock import BedrockModel
from services.storage_service import StorageService
from config import config


class MediaType(Enum):
    IMAGE_GEN = "IMAGE_GENERATION"
    VIDEO_GEN = "VIDEO_GENERATION"
  
    @classmethod
    def from_string(cls, string_value):
        """Convert string value to corresponding enum member"""
        for member in cls:
            if member.value == string_value:
                return member
        return None


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
    ):
        if 'request_history' not in st.session_state:
            st.session_state.request_history = []
        
        storage_metadata = None
        if media_file and prompt:
            try:
                storage_metadata = self.storage_service.upload_image(
                    media_type=media_type.value,
                    model_type=model_type.value,
                    prompt=prompt,
                    details=details,
                    status='Completed',
                    image=media_file,
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
