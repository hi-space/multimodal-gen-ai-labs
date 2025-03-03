import os
import sys

ROOT_PATH = os.path.abspath("../../")
sys.path.append(ROOT_PATH)
import pandas as pd
import os
import base64
import json
import pandas as pd
from functools import wraps
from PIL import Image
from io import BytesIO
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from functools import wraps

import boto3
import sagemaker
import streamlit as st
import time
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

from genai_kit.aws.embedding import BedrockEmbedding
from genai_kit.aws.claude import BedrockClaude
from genai_kit.aws.bedrock import BedrockModel
from genai_kit.utils.images import encode_image_base64, encode_image_base64_from_file, display_image


from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import OpenSearchVectorSearch

# Custom imports
from genai_kit.aws.embedding import BedrockEmbedding
from genai_kit.aws.claude import BedrockClaude
from genai_kit.aws.bedrock import BedrockModel
from genai_kit.utils.images import encode_image_base64, encode_image_base64_from_file, display_image
from genai_kit.aws.amazon_image import BedrockAmazonImage, TitanImageSize, ImageParams, ControlMode, OutpaintMode


from langchain.prompts import PromptTemplate



# 에이전트 유형 정의
class AgentType(Enum):
    PRODUCT_SEARCH = "product_search"
    BACKGROUND_REMOVAL = "background_removal"
    IMAGE_VARIATION = "image_variation"
    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"
    CONTENT_GENERATION = "content_generation"
    GENERAL_CONVERSATION = "general_conversation"


# 의도 분류 에이전트 프롬프트
INTENT_CLASSIFICATION_TEMPLATE = '''
당신은 사용자의 요청을 분석하여 어떤 유형의 작업이 필요한지 정확히 판단하는 전문가입니다.
다음 중 하나의 카테고리를 선택하세요:

1. product_search: 상품을 찾거나 검색하는 요청
2. background_removal: 이미지에서 배경을 제거하는 요청
3. image_variation: 기존 이미지의 변형을 생성하는 요청
4. inpainting: 이미지의 특정 부분을 채우거나 수정하는 요청
5. outpainting: 이미지 외부 영역을 확장하는 요청
6. content_generation: 광고 문구, 상품 설명 등의 텍스트 생성 요청
7. general_conversation: 위 카테고리에 속하지 않는 일반 대화

다음 정보를 바탕으로 판단해주세요:
- 이전 대화 내용
- 현재 사용자 질문
- 현재 표시된 상품 정보

대화 기록:
{conversation_history}

현재 표시된 상품 정보:
{current_products}

사용자 질문: 
{question}

응답 형식:
```
{{
  "agent_type": "여기에 유형 입력",
  "reasoning": "분류 이유 설명",
  "parameters": {{
    // 특정 에이전트에 필요한 매개변수
    // product_search의 경우: "keyword": "영문 검색어"
    // image_variation/inpainting/outpainting의 경우: "image_index": 이미지 번호, "instructions": "구체적인 지시사항"
  }}
}}
```

오직 JSON 형식으로만 응답하세요.
'''

# 이미지 편집 에이전트 프롬프트
IMAGE_EDITING_TEMPLATE = '''
당신은 사용자의 이미지 편집 요청을 이해하고 정확한 편집 매개변수를 제공하는 전문가입니다.

편집 유형: {edit_type}
사용자 요청: {request}
선택된 이미지 번호: {image_index}
이미지 정보: {image_info}

다음 정보를 JSON 형식으로 제공하세요:

```
{{
  "edit_instructions": "영어로 상세한 편집 지시사항",
  "negative_prompt": "영어로 원하지 않는 요소를 설명하는 부정 프롬프트(선택 사항)",
  "additional_parameters": {{
    // 특정 편집 유형에 필요한 매개변수
    // outpainting의 경우 필수 항목: "mask_prompt": "확장하려는 영역 설명", "mode": "DEFAULT 또는 PRECISE"
  }}
}}
```

주의: outpainting 유형의 경우 "additional_parameters" 내에 "mask_prompt"를 반드시 포함해야 합니다.
mask_prompt는 이미지의 어느 부분을 확장할지 설명하는 것입니다. (예: "background", "surrounding area", "bottom area" 등)

오직 JSON 형식으로만 응답하세요.
'''

# 광고 문구 생성 에이전트 프롬프트
CONTENT_GENERATION_TEMPLATE = '''
당신은 상품 이미지나 설명을 바탕으로 효과적인 마케팅 콘텐츠를 생성하는 전문가입니다.

생성 유형: {content_type}
상품 이미지 정보: {image_info}
상품 설명: {product_description}
사용자 요청: {request}

예시 양식:
- 광고 문구인 경우: 짧고 기억에 남는 문구와 함께 3-5문장의 설득력 있는 설명
- 상품 설명인 경우: 특징, 장점, 사용 방법, 타겟 사용자 등을 포함한 300-500자 설명

요청에 맞는 {content_type}을 생성해주세요.
'''

# 일반 대화 에이전트 프롬프트
GENERAL_CONVERSATION_TEMPLATE = '''
당신은 사용자와의 자연스러운 대화를 나누는 친절한 쇼핑 도우미입니다.

대화 기록:
{conversation_history}

사용자 메시지: {message}

상품 정보(참고용):
{product_info}

사용자의 메시지에 자연스럽게 응답하세요. 필요하다면 상품 정보를 참고할 수 있지만, 사용자가 특별히 요청하지 않았다면 상품을 직접적으로 홍보하지 마세요.
'''


class MultimodalAgentSystem:
    def __init__(self, region='us-west-2'):
        self.region = region
        self.initialize_aws_clients()
        self.initialize_models()
        self.initialize_agents()
        self.initialize_session_state()
        
    def initialize_aws_clients(self):
        """AWS 클라이언트 초기화"""
        self.boto3_session = boto3.session.Session(region_name=self.region)
        self.s3_client = self.boto3_session.client('s3')
        
        # OpenSearch 설정 불러오기
        with open("oss_policies_info.json", "r") as f:
            saved_data = json.load(f)
            
        self.host = saved_data["opensearch_host"]
        self.index_name = saved_data["opensearch_index_name"]
        
        # AWS 인증 설정
        credentials = boto3.Session().get_credentials()
        awsauth = AWSV4SignerAuth(credentials, self.region, 'aoss')
        
        # OpenSearch 클라이언트 초기화
        self.oss_client = OpenSearch(
            hosts=[{'host': self.host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=600
        )
    
    def initialize_models(self):
        """모델 초기화"""
        self.claude = BedrockClaude(region=self.region, modelId=BedrockModel.SONNET_3_5_CR)
        self.bedrock_embedding = BedrockEmbedding(region=self.region)
        
        # Langchain 모델 초기화
        self.llm = self.claude.get_chat_model()
        
        # 이미지 생성 모델 초기화
        self.image_generator = BedrockAmazonImage(region=self.region)
    

    def initialize_agents(self):
        """에이전트 초기화"""
        # 의도 분류 에이전트
        self.intent_classifier = (
            PromptTemplate.from_template(INTENT_CLASSIFICATION_TEMPLATE) 
            | self.llm 
            | JsonOutputParser()
        )
        
        # 이미지 편집 에이전트
        self.image_editing_agent = (
            PromptTemplate.from_template(IMAGE_EDITING_TEMPLATE) 
            | self.llm 
            | JsonOutputParser()
        )
        
        # 이미지 편집 후처리 함수
        def process_edit_params(edit_params, edit_type):
            """편집 파라미터 후처리 및 유효성 검증"""
            if edit_type == 'outpainting':
                # outpainting은 mask_prompt가 필수
                if 'additional_parameters' not in edit_params:
                    edit_params['additional_parameters'] = {}
                
                if 'mask_prompt' not in edit_params['additional_parameters']:
                    edit_params['additional_parameters']['mask_prompt'] = "background, surrounding area"
            
            return edit_params
        
        # 후처리 함수를 포함한 이미지 편집 에이전트 재정의
        self.image_editing_agent_with_processing = lambda params: process_edit_params(
            self.image_editing_agent.invoke(params), 
            params['edit_type']
        )
        
        # 콘텐츠 생성 에이전트
        self.content_generation_agent = (
            PromptTemplate.from_template(CONTENT_GENERATION_TEMPLATE) 
            | self.llm 
            | StrOutputParser()
        )
        
        # 일반 대화 에이전트
        self.general_conversation_agent = (
            PromptTemplate.from_template(GENERAL_CONVERSATION_TEMPLATE) 
            | self.llm 
            | StrOutputParser()
        )
    
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "search_results" not in st.session_state:
            st.session_state.search_results = []
        if "current_image" not in st.session_state:
            st.session_state.current_image = None
        if "edited_images" not in st.session_state:
            st.session_state.edited_images = []
        if "conversation_results" not in st.session_state:
            st.session_state.conversation_results = {}
    
    def get_conversation_history(self, max_messages=10):
        """대화 기록 가져오기"""
        recent_messages = st.session_state.messages[-max_messages:] if len(st.session_state.messages) > 0 else []
        return "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in recent_messages
        ])
    
    def get_current_products_info(self):
        """현재 표시된 상품 정보 가져오기"""
        if not st.session_state.search_results:
            return "현재 표시된 상품이 없습니다."
        
        products_info = []
        for idx, product in enumerate(st.session_state.search_results):
            products_info.append(f"상품 {idx+1}: {product['item_name']} - {product['description']}")
        
        return "\n".join(products_info)
    
    def find_similar_items(self, text=None, image=None, k=5):
        """유사 상품 검색"""
        query_emb = self.bedrock_embedding.embedding_multimodal(text=text, image=image)

        body = {
            "size": k,
            "_source": {
                "exclude": ["image_vector"],
            },
            "query": {
                "knn": {
                    "image_vector": {
                        "vector": query_emb,
                        "k": k,
                    }
                }
            },
        }

        res = self.oss_client.search(index=self.index_name, body=body)
        return res["hits"]["hits"]
    
    def process_search_results(self, documents):
        """검색 결과 처리"""
        data = []
        for _, doc in enumerate(documents):
            source = doc.get('_source', {})
            metadata = source.get('metadata', {})
            score = doc.get('_score', 0)

            try:
                img_res = self.s3_client.get_object(
                    Bucket="amazon-berkeley-objects",
                    Key=f"images/small/{metadata.get('image_url', '')}"
                )
                img = Image.open(BytesIO(img_res['Body'].read()))
                img_base64 = encode_image_base64(img) if img else ''
            except Exception as e:
                st.error(f"이미지 로드 오류: {str(e)}")
                img = None
                img_base64 = ''

            data.append({
                'image': img,
                'image_base64': img_base64,
                'item_name': metadata.get('item_name', ''),
                'item_id': metadata.get('item_id', ''),
                'image_url': metadata.get('image_url', ''),
                'description': source.get('description', ''),
                'score': score
            })
        
        return data
    
    def handle_product_search(self, parameters):
        """상품 검색 처리"""
        keyword = parameters.get('keyword', '')
        st.info(f"'{keyword}' 관련 상품을 검색합니다...")
        
        # 상품 검색 실행
        docs = self.find_similar_items(text=keyword, image=st.session_state.get("current_image"))
        
        # 검색 결과 처리
        new_results = self.process_search_results(docs)
        
        # 세션 상태에 검색 결과 저장
        st.session_state.search_results = new_results
        
        # 검색 결과 표시
        with st.container():
            st.subheader("상품 검색 결과")
            cols = st.columns(3)
            for idx, result in enumerate(new_results):
                with cols[idx % 3]:
                    if result['image'] is not None:
                        st.image(result['image'], caption=f"{idx+1}. {result['item_name']}")
                    else:
                        st.error(f"{idx+1}. {result['item_name']} - 이미지 로드 실패")
                    st.write(result['description'][:100] + "..." if len(result['description']) > 100 else result['description'])
        
        # 응답 생성
        response = f"'{keyword}' 관련 상품을 {len(new_results)}개 찾았습니다. 원하시는 상품이 있으면 번호를 알려주세요."
        return response
    
    def handle_background_removal(self, parameters):
        """배경 제거 처리"""
        image_index = parameters.get('image_index', 1) - 1
        
        if image_index < 0 or image_index >= len(st.session_state.search_results):
            return f"유효하지 않은 이미지 번호입니다. 1부터 {len(st.session_state.search_results)}까지의 번호를 입력해주세요."
        
        st.info(f"{image_index+1}번 상품 이미지의 배경을 제거합니다...")
        
        # 선택된 이미지 가져오기
        selected_image = st.session_state.search_results[image_index]
        image_base64 = selected_image['image_base64']
        
        try:
            # 이미지 파라미터 설정
            image_params = ImageParams()
            body = image_params.background_removal(image=image_base64)
            
            # 배경 제거 실행
            result_images = self.image_generator.generate_image(body)
            
            if result_images and len(result_images) > 0:
                # 이미지 디코딩 및 표시
                result_image_data = base64.b64decode(result_images[0])
                result_image = Image.open(BytesIO(result_image_data))
                result_image_base64 = encode_image_base64(result_image)
                
                # 결과 이미지 저장
                edited_image_info = {
                    'type': 'background_removal',
                    'original_index': image_index,
                    'original_name': selected_image['item_name'],
                    'image': result_image,
                    'image_base64': result_image_base64,
                    'timestamp': time.time()
                }
                st.session_state.edited_images.append(edited_image_info)
                
                # 이미지 표시
                st.image(result_image, caption=f"{selected_image['item_name']} - 배경 제거됨")
                
                return f"{image_index+1}번 상품의 배경을 성공적으로 제거했습니다."
            else:
                return "배경 제거 중 오류가 발생했습니다. 다른 이미지를 시도해보세요."
        
        except Exception as e:
            st.error(f"배경 제거 오류: {str(e)}")
            return f"배경 제거 중 오류가 발생했습니다: {str(e)}"
    
    def handle_image_variation(self, parameters):
        """이미지 변형 처리"""
        image_index = parameters.get('image_index', 1) - 1
        instructions = parameters.get('instructions', '')
        
        if image_index < 0 or image_index >= len(st.session_state.search_results):
            return f"유효하지 않은 이미지 번호입니다. 1부터 {len(st.session_state.search_results)}까지의 번호를 입력해주세요."
        
        # 이미지 편집 매개변수 추출
        edit_params = self.image_editing_agent.invoke({
            'edit_type': 'image_variation',
            'request': instructions,
            'image_index': image_index + 1,
            'image_info': st.session_state.search_results[image_index]['item_name']
        })
        
        st.info(f"{image_index+1}번 상품 이미지의 변형을 생성합니다...")
        
        # 선택된 이미지 가져오기
        selected_image = st.session_state.search_results[image_index]
        image_base64 = selected_image['image_base64']
        
        try:
            # 이미지 파라미터 설정
            image_params = ImageParams()
            body = image_params.image_variant(
                images=[image_base64],
                text=edit_params.get('edit_instructions', instructions),
                negative_text=edit_params.get('negative_prompt'),
                similarity=0.7
            )
            
            # 이미지 변형 실행
            result_images = self.image_generator.generate_image(body)
            
            if result_images and len(result_images) > 0:
                # 이미지 디코딩 및 표시
                result_image_data = base64.b64decode(result_images[0])
                result_image = Image.open(BytesIO(result_image_data))
                result_image_base64 = encode_image_base64(result_image)
                
                # 결과 이미지 저장
                edited_image_info = {
                    'type': 'image_variation',
                    'original_index': image_index,
                    'original_name': selected_image['item_name'],
                    'image': result_image,
                    'image_base64': result_image_base64,
                    'instructions': edit_params.get('edit_instructions', instructions),
                    'timestamp': time.time()
                }
                st.session_state.edited_images.append(edited_image_info)
                
                # 이미지 표시
                st.image(result_image, caption=f"{selected_image['item_name']} - 변형됨")
                
                return f"{image_index+1}번 상품의 변형 이미지를 생성했습니다."
            else:
                return "이미지 변형 중 오류가 발생했습니다. 다른 이미지를 시도해보세요."
        
        except Exception as e:
            st.error(f"이미지 변형 오류: {str(e)}")
            return f"이미지 변형 중 오류가 발생했습니다: {str(e)}"
    
    def handle_inpainting(self, parameters):
        """인페인팅 처리"""
        image_index = parameters.get('image_index', 1) - 1
        instructions = parameters.get('instructions', '')
        
        if image_index < 0 or image_index >= len(st.session_state.search_results):
            return f"유효하지 않은 이미지 번호입니다. 1부터 {len(st.session_state.search_results)}까지의 번호를 입력해주세요."
        
        # 이미지 편집 매개변수 추출
        edit_params = self.image_editing_agent.invoke({
            'edit_type': 'inpainting',
            'request': instructions,
            'image_index': image_index + 1,
            'image_info': st.session_state.search_results[image_index]['item_name']
        })
        
        st.info(f"{image_index+1}번 상품 이미지의 특정 부분을 수정합니다...")
        
        # 선택된 이미지 가져오기
        selected_image = st.session_state.search_results[image_index]
        image_base64 = selected_image['image_base64']
        
        try:
            # 이미지 파라미터 설정
            image_params = ImageParams()
            body = image_params.inpainting(
                image=image_base64,
                text=edit_params.get('edit_instructions', instructions),
                mask_prompt=edit_params.get('additional_parameters', {}).get('mask_prompt', '변경할 부분'),
                negative_text=edit_params.get('negative_prompt')
            )
            
            # 인페인팅 실행
            result_images = self.image_generator.generate_image(body)
            
            if result_images and len(result_images) > 0:
                # 이미지 디코딩 및 표시
                result_image_data = base64.b64decode(result_images[0])
                result_image = Image.open(BytesIO(result_image_data))
                result_image_base64 = encode_image_base64(result_image)
                
                # 결과 이미지 저장
                edited_image_info = {
                    'type': 'inpainting',
                    'original_index': image_index,
                    'original_name': selected_image['item_name'],
                    'image': result_image,
                    'image_base64': result_image_base64,
                    'instructions': edit_params.get('edit_instructions', instructions),
                    'timestamp': time.time()
                }
                st.session_state.edited_images.append(edited_image_info)
                
                # 이미지 표시
                st.image(result_image, caption=f"{selected_image['item_name']} - 인페인팅 적용됨")
                
                return f"{image_index+1}번 상품의 인페인팅을 성공적으로 적용했습니다."
            else:
                return "인페인팅 중 오류가 발생했습니다. 다른 이미지를 시도해보세요."
        
        except Exception as e:
            st.error(f"인페인팅 오류: {str(e)}")
            return f"인페인팅 중 오류가 발생했습니다: {str(e)}"
    
    def handle_outpainting(self, parameters):
        """아웃페인팅 처리"""
        image_index = parameters.get('image_index', 1) - 1
        instructions = parameters.get('instructions', '')
        
        if image_index < 0 or image_index >= len(st.session_state.search_results):
            return f"유효하지 않은 이미지 번호입니다. 1부터 {len(st.session_state.search_results)}까지의 번호를 입력해주세요."
        
        # 이미지 편집 매개변수 추출 (후처리 포함)
        edit_params = self.image_editing_agent_with_processing({
            'edit_type': 'outpainting',
            'request': instructions,
            'image_index': image_index + 1,
            'image_info': st.session_state.search_results[image_index]['item_name']
        })
        
        st.info(f"{image_index+1}번 상품 이미지를 확장합니다...")
        
        # 선택된 이미지 가져오기
        selected_image = st.session_state.search_results[image_index]
        image_base64 = selected_image['image_base64']
        
        try:
            # 모드 결정
            mode_str = edit_params.get('additional_parameters', {}).get('mode', 'DEFAULT')
            mode = OutpaintMode.PRECISE if mode_str.upper() == 'PRECISE' else OutpaintMode.DEFAULT
            
            # mask_prompt 값 확인 - 필수이므로 기본값 설정
            mask_prompt = edit_params.get('additional_parameters', {}).get('mask_prompt')
            if not mask_prompt:
                # 기본 마스크 프롬프트 설정 - 배경 확장에 적합한 값
                mask_prompt = "background, surrounding area"
            
            # 이미지 파라미터 설정
            image_params = ImageParams()
            body = image_params.outpainting(
                image=image_base64,
                text=edit_params.get('edit_instructions', instructions),
                mask_prompt=mask_prompt,  # 기본값이 있는 마스크 프롬프트 사용
                mode=mode,
                negative_text=edit_params.get('negative_prompt')
            )
            
            # 아웃페인팅 실행
            result_images = self.image_generator.generate_image(body)
            
            if result_images and len(result_images) > 0:
                # 이미지 디코딩 및 표시
                result_image_data = base64.b64decode(result_images[0])
                result_image = Image.open(BytesIO(result_image_data))
                result_image_base64 = encode_image_base64(result_image)
                
                # 결과 이미지 저장
                edited_image_info = {
                    'type': 'outpainting',
                    'original_index': image_index,
                    'original_name': selected_image['item_name'],
                    'image': result_image,
                    'image_base64': result_image_base64,
                    'instructions': edit_params.get('edit_instructions', instructions),
                    'timestamp': time.time()
                }
                st.session_state.edited_images.append(edited_image_info)
                
                # 이미지 표시
                st.image(result_image, caption=f"{selected_image['item_name']} - 확장됨")
                
                return f"{image_index+1}번 상품의 이미지를 확장했습니다."
            else:
                return "아웃페인팅 중 오류가 발생했습니다. 다른 이미지를 시도해보세요."
        
        except Exception as e:
            st.error(f"아웃페인팅 오류: {str(e)}")
            return f"아웃페인팅 중 오류가 발생했습니다: {str(e)}"
    
    def handle_content_generation(self, parameters):
        """콘텐츠 생성 처리"""
        content_type = parameters.get('content_type', 'ad_copy')
        image_index = parameters.get('image_index', -1)
        
        # 이미지 정보 및 제품 설명 준비
        if image_index >= 0 and image_index < len(st.session_state.search_results):
            image_info = st.session_state.search_results[image_index]['item_name']
            product_description = st.session_state.search_results[image_index]['description']
        elif len(st.session_state.edited_images) > 0:
            # 가장 최근에 편집된 이미지 사용
            latest_edited = st.session_state.edited_images[-1]
            image_info = f"{latest_edited['original_name']} (편집됨: {latest_edited['type']})"
            
            if latest_edited['original_index'] >= 0 and latest_edited['original_index'] < len(st.session_state.search_results):
                product_description = st.session_state.search_results[latest_edited['original_index']]['description']
            else:
                product_description = "제품 설명 없음"
        else:
            image_info = "사용 가능한 이미지 없음"
            product_description = "제품 설명 없음"
        
        st.info(f"{content_type} 콘텐츠를 생성 중입니다...")
        
        # 콘텐츠 생성
        generated_content = self.content_generation_agent.invoke({
            'content_type': '광고 문구' if content_type == 'ad_copy' else '상품 설명',
            'image_info': image_info,
            'product_description': product_description,
            'request': parameters.get('instructions', '효과적인 콘텐츠를 생성해주세요.')
        })
        
        return generated_content
    
    def handle_general_conversation(self, message):
        """일반 대화 처리"""
        conversation_history = self.get_conversation_history()
        product_info = self.get_current_products_info()
        
        response = self.general_conversation_agent.invoke({
            'conversation_history': conversation_history,
            'message': message,
            'product_info': product_info
        })
        
        return response
    
    def process_message(self, message, uploaded_file=None):
        """메시지 처리 메인 함수"""
        # 고유 메시지 ID 생성
        message_id = f"msg_{int(time.time() * 1000)}"
        
        # 대화별 결과 딕셔너리 업데이트
        if message_id not in st.session_state.conversation_results:
            st.session_state.conversation_results[message_id] = {
                "message": message if message else "Image upload",
                "intent_classification": None,
                "response": None
            }
        
        # 업로드된 이미지 처리
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.current_image = encode_image_base64(image)
        
        # 메시지 처리
        if message or uploaded_file:
            # 대화 기록 및 상품 정보 준비
            conversation_history = self.get_conversation_history()
            current_products = self.get_current_products_info()
            
            # 의도 분류
            try:
                intent_classification = self.intent_classifier.invoke({
                    'conversation_history': conversation_history,
                    'current_products': current_products,
                    'question': message if message else "이미지 분석 및 관련 상품 검색"
                })
                
                # 의도 분류 결과 저장
                st.session_state.conversation_results[message_id]["intent_classification"] = intent_classification
                
                # 분류된 의도에 따라 적절한 에이전트 호출
                agent_type = intent_classification.get('agent_type', 'general_conversation')
                parameters = intent_classification.get('parameters', {})
                
                # 디버깅 정보 출력
                st.write(f"분류된 의도: {agent_type}")
                st.write(f"파라미터: {parameters}")
                
                # 문자열 비교로 변경하여 에러 가능성 줄이기
                if agent_type == "product_search":
                    response = self.handle_product_search(parameters)
                elif agent_type == "background_removal":
                    response = self.handle_background_removal(parameters)
                elif agent_type == "image_variation":
                    response = self.handle_image_variation(parameters)
                elif agent_type == "inpainting":
                    response = self.handle_inpainting(parameters)
                elif agent_type == "outpainting":
                    response = self.handle_outpainting(parameters)
                elif agent_type == "content_generation":
                    response = self.handle_content_generation(parameters)
                else:
                    response = self.handle_general_conversation(message)
                
                # 응답 저장
                st.session_state.conversation_results[message_id]["response"] = response
                
                return response, agent_type
            
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                st.error(f"메시지 처리 중 오류 발생: {str(e)}")
                st.write("상세 오류 정보:")
                st.code(error_trace)
                
                # 기본 응답 제공
                return f"죄송합니다. 요청을 처리하는 중 오류가 발생했습니다. 다른 질문을 시도해 주세요.", "error"
                

def main():
    st.title("이미지 편집 기능이 강화된 상품 검색 챗봇")
    
    # 멀티모달 에이전트 시스템 초기화
    agent_system = MultimodalAgentSystem()
    
    # 메시지 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 이미지가 있는 경우 표시
            if 'image' in message:
                st.image(message['image'])
    
    # 입력 섹션
    # uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])
    uploaded_file = None
    message = st.chat_input("메시지를 입력하세요...")
    
    if message or uploaded_file:
        # 사용자 메시지 즉시 표시
        if message:
            st.session_state.messages.append({
                "role": "user", 
                "content": message
            })
            with st.chat_message("user"):
                st.markdown(message)
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.messages.append({
                "role": "user", 
                "content": "이미지를 업로드했습니다",
                "image": image
            })
            with st.chat_message("user"):
                st.markdown("이미지를 업로드했습니다")
                st.image(image)
        
        # 메시지 처리
        with st.spinner('처리 중...'):
            response, agent_type = agent_system.process_message(message, uploaded_file)
            
            # 응답 표시
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "agent_type": agent_type
            })
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()