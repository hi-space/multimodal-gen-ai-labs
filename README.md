# Multimodal Generative AI

This repository focuses on developing multimodal generative AI applications by leveraging various AWS services. It integrates various cutting-edge technologies and models to process and generate text, images, and other data types, offering a range of functionalities.

- **Multimodal LLM**: Enhances language understanding and generation by combining text and image data. It uses [Amazon Bedrock's Claude](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html).
  
- **Multimodal Embedding**: Represents text and images in a unified vector space, allowing for similarity comparison and retrieval between data. It uses the [Amazon Titan Text Embeddings v2](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html) and the [Amazon Titan Multimodal Embeddings G1 model](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html).
  
- **Multimodal RAG (Retrieval-Augmented Generation)**: Receives text or images as input to retrieve relevant data and generates answers. It uses [Amazon Bedrock's Claude](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html), [Amazon Titan Text Embeddings v2](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html), [Amazon Titan Multimodal Embeddings G1 model](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html), and [Amazon OpenSearch Service](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/what-is.html) as a vectorDB.

- **Image Generation with Multimodal LLM**: Generates high-quality images based on textual or multimodal inputs. It uses [Amazon Titan Image Generator G1 model](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-image-models.html) and [Amazon Bedrock's Claude](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html).

This project demonstrates how AWS services are used to create, manage, and scale these multimodal AI capabilities. Whether you’re building research tools, creative applications, or advanced AI solutions, this repository serves as a comprehensive starting point.
