# Multimodal Embedding Search

OpenSearch Serverless와 Bedrock Titan Multimodal Embedding 모델을 활용합니다.

### `1-preparation.ipynb`

필요한 리소스를 준비하고 샘플 데이터를 삽입하는 노트북입니다. 전체 작업이 수행되는 데 약 20분 소요됩니다.
 - OpenSearch Serverless (VectorDB)의 Collection, index 를 생성
 - 샘플 데이터를 Multimodal Embedding 하여 document 삽입

### `2-multimodal-search.ipynb`

VectorDB에 저장된 데이터를 기반으로 semantic 검색 테스트를 합니다. 이미지 또는 텍스트로 검색을 수행할 수 있고, 쿼리와 유사한 데이터들을 확인할 수 있습니다.

### 텍스트 검색

| TEST-1                                     | TEST-2                                     |
|--------------------------------------------|--------------------------------------------|
| ![text-query-1](./assets/text-query-1.png) | ![text-query-2](./assets/text-query-2.png) |

### 이미지 검색

| TEST-1                                       | TEST-2                                       |
|----------------------------------------------|----------------------------------------------|
| ![image-query-1](./assets/image-query-1.png) | ![image-query-2](./assets/image-query-2.png) |

### LMM로 이미지 해석 -> 텍스트 검색

| TEST-1                               |
|--------------------------------------|
| ![lmm-query](./assets/lmm-query.png) |

### 상품 검색 챗봇

| TEST-1                         | TEST-2                         |
|--------------------------------|--------------------------------|
| ![chat-1](./assets/chat-1.png) | ![chat-2](./assets/chat-2.png) |

### `3-clean-up.ipynb`

생성한 OpenSearch 리소스들을 삭제합니다.

---

# Streamlit App

## Getting Started

```sh
streamlit run app.py
```

## Architecture

![multimodal-gen-architecture](./assets/multimodal-ad-gen-architecture.png)

## Screenshtop
![multimodal-ad-gen](./assets/multimodal-ad-gen.png)

