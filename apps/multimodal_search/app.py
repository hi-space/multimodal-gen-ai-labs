import os
import sys

ROOT_PATH = os.path.abspath("../../")
sys.path.append(ROOT_PATH)
import pandas as pd
import os
import json
import pandas as pd
from functools import wraps
from PIL import Image
from io import BytesIO

import boto3
import sagemaker
import streamlit as st
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

from genai_kit.aws.embedding import BedrockEmbedding
from genai_kit.aws.claude import BedrockClaude
from genai_kit.aws.bedrock import BedrockModel
from genai_kit.utils.images import encode_image_base64, encode_image_base64_from_file, display_image

from langchain.prompts import PromptTemplate

# 수정된 템플릿: 전체 대화 맥락을 고려하여 키워드 추출
TASK_CLASSIFICATION_TEMPLATE = '''
You are an expert in determining if a user's query requires product search or just a normal conversation.
If product search is needed, suggest an appropriate English search keyword.
The output should be formatted as a JSON instance with these keys:
- "requires_search": boolean indicating if product search is needed
- "keyword": English search product keyword (only when requires_search is true)

Just answer json without any other explanation.

Here is the conversation history with the user:
<conversation_history>
{conversation_history}
</conversation_history>

Current question from user:
<question>
{question}
</question>
'''

PRODUCT_SEARCH_TEMPLATE = """
You are an expert who finds the products users want. Based on the user's question in <question>, refer to the current search results, previous search results, and conversation history to recommend a product to the user.

Previous conversations:
<conversations>
{conversations}
</conversations>

Current search results (if any):
<current_results>
{information}
</current_results>

Previous search results:
<previous_results>
{previous_results}
</previous_results>

Here is a question from user:
<question>
{question}
</question>

Instructions:
1. If the user asks about a specific numbered item (e.g., "3번 제품"), check if there are any current search results first.
2. If there are no current search results but the user is asking about a specific item, respond with: "죄송합니다만, 질문의 맥락이 부족해서 정확한 답변을 드리기 어렵습니다. \"3번\"이 무엇을 가리키는지 명확하지 않습니다."
3. Otherwise, provide relevant product information based on the available search results.
"""

# Initialize Claude
claude = BedrockClaude(region='us-west-2')

# Load configuration
with open("oss_policies_info.json", "r") as f:
    saved_data = json.load(f)

host = saved_data["opensearch_host"]
index_name = saved_data["opensearch_index_name"]
vector_store_name = saved_data["vector_store_name"]
encryption_policy = saved_data["encryption_policy"]
network_policy = saved_data["network_policy"]
access_policy = saved_data["access_policy"]

# Initialize AWS services
boto3_session = boto3.session.Session(region_name='us-west-2')
sts_client = boto3_session.client('sts')
s3_client = boto3_session.client('s3')
opensearchservice_client = boto3_session.client('opensearchserverless')

service = 'aoss'
credentials = boto3.Session().get_credentials()
awsauth = AWSV4SignerAuth(credentials, boto3_session.region_name, service)

sagemaker_role_arn = sagemaker.get_execution_role()
bedrock_embedding = BedrockEmbedding(region=boto3_session.region_name)

oss_client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=600
)

def show_search_results(documents):
    data = []
    for _, doc in enumerate(documents):
        source = doc.get('_source', {})
        metadata = source.get('metadata', {})
        score = doc.get('_score', 0)

        img_res = s3_client.get_object(
            Bucket="amazon-berkeley-objects",
            Key=f"images/small/{metadata.get('image_url', '')}"
        )
                
        img = Image.open(BytesIO(img_res['Body'].read()))
        img_base64 = encode_image_base64(img) if img else ''

        data.append({
            'image': img,
            'item_name': metadata.get('item_name', ''),
            'item_id': metadata.get('item_id', ''),
            'image_url': metadata.get('image_url', ''),
            'description': source.get('description', ''),
            'score': score
        })
    
    return data

def find_similar_items(oss_client,
                      bedrock_embedding: BedrockEmbedding,
                      index_name: str,
                      k: int = 5,
                      text: str = None,
                      image: str = None):
    query_emb = bedrock_embedding.embedding_multimodal(text=text, image=image)

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

    res = oss_client.search(index=index_name, body=body)
    return res["hits"]["hits"]

def process_message(message, uploaded_file):
    # 새 검색 결과와 의도 분류 저장을 위한 세션 상태 초기화
    if 'all_search_results' not in st.session_state:
        st.session_state.all_search_results = []
    if 'all_intent_classifications' not in st.session_state:
        st.session_state.all_intent_classifications = []
    
    # 현재 메시지에 대한 고유 ID 생성 (타임스탬프 기반)
    import time
    message_id = f"msg_{int(time.time() * 1000)}"
    
    # 대화별 결과 딕셔너리에 현재 메시지를 위한 공간 생성
    if message_id not in st.session_state.conversation_results:
        st.session_state.conversation_results[message_id] = {
            "message": message if message else "Image upload",
            "intent_classification": None,
            "search_results": None,
            "response": None
        }
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.current_image = encode_image_base64(image)

    if message or uploaded_file:
        # 전체 대화 내용 준비
        conversation_history = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in st.session_state.messages[-10:]  # 최근 10개 메시지
        ])
        
        # 메시지의 의도 분류
        task_info = {"requires_search": True}  # 기본값 설정
        try:
            task_classification = claude.invoke_llm_response(
                text=PromptTemplate(
                    template=TASK_CLASSIFICATION_TEMPLATE,
                    input_variables=["question", "conversation_history"]).format(
                        question=message if message else "관련 상품 추천해줘",
                        conversation_history=conversation_history
                    ),
                image=st.session_state.get("current_image")
            )
            task_info = json.loads(task_classification)
            
            # 의도 분류 결과 저장
            intent_data = {
                "message": message if message else "Image upload",
                "classification": task_info
            }
            st.session_state.all_intent_classifications.append(intent_data)
            
            # 현재 메시지의 대화별 결과에 의도 분류 저장
            st.session_state.conversation_results[message_id]["intent_classification"] = task_info
            
            # 현재 대화에 대한 의도 분류 결과 표시
            with st.chat_message("assistant"):
                with st.expander("Intent Classification", expanded=False):
                    st.json(task_info)
                    
                # 의도 분류 정보가 대화 흐름에 포함되도록 함
                intent_message = {
                    "role": "assistant", 
                    "content": f"Intent Classification: {json.dumps(task_info, ensure_ascii=False)}",
                    "type": "intent_classification",
                    "message_id": message_id
                }
                st.session_state.messages.append(intent_message)
            
        except Exception as e:
            st.error(f"의도 분류 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본값으로 검색 수행
            task_info = {"requires_search": True}
            
            # 오류 정보를 대화 흐름에 포함
            with st.chat_message("assistant"):
                with st.expander("Error in Intent Classification", expanded=True):
                    st.error(f"의도 분류 중 오류가 발생했습니다: {str(e)}")
                
                # 오류 메시지 추가
                error_message = {
                    "role": "assistant", 
                    "content": f"Error in intent classification. Using default search mode.",
                    "type": "error",
                    "message_id": message_id
                }
                st.session_state.messages.append(error_message)
        
        requires_search = task_info.get('requires_search', True)

        if requires_search:
            # 의도 분류에서 키워드 사용
            search_keyword = task_info.get('keyword', message if message else "")

            # 유사 아이템 검색
            docs = find_similar_items(
                oss_client,
                bedrock_embedding,
                index_name=index_name,
                text=search_keyword,
                image=st.session_state.get("current_image")
            )

            # 검색 결과 표시
            new_results = show_search_results(docs)
            
            # 세션 상태에 검색 결과 저장
            if 'search_results' not in st.session_state:
                st.session_state.search_results = []
            
            # 중복되지 않는 결과만 추가
            for new_r in new_results:
                if not any(r['item_id'] == new_r['item_id'] for r in st.session_state.search_results):
                    st.session_state.search_results.append(new_r)
            
            # 현재 메시지와 결과를 all_search_results에 저장
            st.session_state.all_search_results.append({
                "message": message if message else "Image upload",
                "results": new_results
            })
            
            # 이전 검색 결과 준비
            previous_results = "\n".join([
                f"- {result['item_name']}: {result['description']}"
                for result in st.session_state.search_results
            ])

            text = PromptTemplate(
                template=PRODUCT_SEARCH_TEMPLATE,
                input_variables=["question", "information", "conversations", "previous_results"]
            ).format(
                question=message if message else "관련 상품 추천해줘",
                information=str(docs),
                conversations=conversation_history,
                previous_results=previous_results
            )

            # 현재 메시지와 결과를 all_search_results에 저장
            search_data = {
                "message": message if message else "Image upload",
                "results": new_results
            }
            st.session_state.all_search_results.append(search_data)
            
            # 응답 스트리밍을 위한 플레이스홀더 생성
            with st.chat_message("assistant"):
                # 먼저 검색 결과를 expander에 표시
                with st.expander("Current Search Results", expanded=True):
                    cols = st.columns(3)
                    for idx, result in enumerate(new_results):
                        with cols[idx % 3]:
                            st.image(result['image'], caption=result['item_name'])
                
                # 검색 결과 정보를 대화 메시지로 추가
                search_result_summary = f"Search Results ({len(new_results)} items):\n" + "\n".join([
                    f"- {result['item_name']}" for result in new_results
                ])
                
                # 검색 결과 메시지 추가
                search_message = {
                    "role": "assistant", 
                    "content": search_result_summary,
                    "type": "search_results",
                    "message_id": message_id
                }
                st.session_state.messages.append(search_message)
                
                # 현재 메시지의 대화별 결과에 검색 결과 저장
                st.session_state.conversation_results[message_id]["search_results"] = new_results
                
                # 응답 스트리밍 표시
                message_placeholder = st.empty()
                full_response = ""
                
                # 응답 스트리밍
                for chunk in claude.converse_stream(
                    text=text,
                    image=st.session_state.get("current_image")
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                # 최종 응답 업데이트 (커서 없이)
                message_placeholder.markdown(full_response)

                print(full_response)
            
            # 전체 응답 저장
            response_message = {
                "role": "assistant", 
                "content": full_response,
                "type": "final_response",
                "message_id": message_id
            }
            st.session_state.messages.append(response_message)
            
            # 현재 메시지의 대화별 결과에 최종 응답 저장
            st.session_state.conversation_results[message_id]["response"] = full_response

        else:
            # 검색이 필요없는 쿼리의 경우에도 동일한 템플릿 사용하여 컨텍스트 유지
            with st.chat_message("assistant"):
                # 일반 대화 메시지임을 표시
                with st.expander("Regular Conversation", expanded=False):
                    st.info("이 메시지는 상품 검색이 필요 없는 일반 대화입니다.")
                
                # 일반 대화임을 대화 메시지로 추가
                regular_message = {
                    "role": "assistant", 
                    "content": "Regular Conversation (No product search required)",
                    "type": "regular_conversation",
                    "message_id": message_id
                }
                st.session_state.messages.append(regular_message)
                
                message_placeholder = st.empty()
                full_response = ""
                
                # 이전 검색 결과 준비
                previous_results = "\n".join([
                    f"- {result['item_name']}: {result['description']}"
                    for result in st.session_state.search_results
                ])

                text = PromptTemplate(
                    template=PRODUCT_SEARCH_TEMPLATE,
                    input_variables=["question", "information", "conversations", "previous_results"]
                ).format(
                    question=message,
                    information="[]",  # 현재 검색 결과 없음
                    conversations=conversation_history,
                    previous_results=previous_results
                )
                
                # 응답 스트리밍
                for chunk in claude.converse_stream(
                    text=text,
                    image=st.session_state.get("current_image")
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                # 최종 응답 업데이트 (커서 없이)
                message_placeholder.markdown(full_response)
            
            # 전체 응답 저장
            response_message = {
                "role": "assistant", 
                "content": full_response,
                "type": "final_response",
                "message_id": message_id
            }
            st.session_state.messages.append(response_message)
            
            # 현재 메시지의 대화별 결과에 최종 응답 저장
            st.session_state.conversation_results[message_id]["response"] = full_response
            # 기존 검색 결과 유지

def main():
    st.title("Multimodal Product Search")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "all_search_results" not in st.session_state:
        st.session_state.all_search_results = []
    if "all_intent_classifications" not in st.session_state:
        st.session_state.all_intent_classifications = []
    # 대화별 결과를 저장하는 딕셔너리
    if "conversation_results" not in st.session_state:
        st.session_state.conversation_results = {}

    # 채팅 인터페이스 - 메시지 타입에 따라 다르게 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # 메시지 타입 확인
            msg_type = message.get("type", "text")
            msg_id = message.get("message_id", None)
            
            if msg_type == "intent_classification" and msg_id:
                # 의도 분류 결과 표시
                with st.expander("Intent Classification", expanded=False):
                    intent_data = st.session_state.conversation_results.get(msg_id, {}).get("intent_classification", {})
                    st.json(intent_data)
                st.markdown(message["content"])
            
            elif msg_type == "search_results" and msg_id:
                # 검색 결과 표시
                search_results = st.session_state.conversation_results.get(msg_id, {}).get("search_results", [])
                if search_results:
                    with st.expander("Search Results", expanded=True):
                        cols = st.columns(3)
                        for idx, result in enumerate(search_results):
                            with cols[idx % 3]:
                                st.image(result['image'], caption=result['item_name'])
                st.markdown(message["content"])
            
            else:
                # 일반 텍스트 메시지
                st.markdown(message["content"])

    # 입력 섹션
    # uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    uploaded_file = None
    message = st.chat_input("Enter your message...")

    if message or uploaded_file:
        # 사용자 메시지 즉시 표시
        if message:
            user_message = {
                "role": "user", 
                "content": message,
                "type": "user_message"
            }
            st.session_state.messages.append(user_message)
            with st.chat_message("user"):
                st.markdown(message)
        if uploaded_file:
            user_image = {
                "role": "user", 
                "content": "Uploaded an image",
                "type": "user_image"
            }
            st.session_state.messages.append(user_image)
            with st.chat_message("user"):
                st.markdown("Uploaded an image")
        
        with st.spinner('Searching for products...'):
            process_message(message, uploaded_file)

if __name__ == "__main__":
    main()