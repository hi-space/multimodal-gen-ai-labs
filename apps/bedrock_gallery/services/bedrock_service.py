import boto3
import re
import json
from typing import List, Optional, Dict, Any
from langchain.prompts import PromptTemplate
from config import config
from genai_kit.aws.claude import BedrockClaude
from genai_kit.aws.bedrock import BedrockModel
from genai_kit.aws.amazon_image import BedrockAmazonImage, ImageParams


def gen_english(request: str,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                top_k: Optional[int] = None):
    prompt = PromptTemplate(
                template="""You are an Assistant for translation.
                Always change the contents in <request> to English without any additional descriptions or tags.
                If there is no <request>, please suggest a new keyword about prop, animal, object or scene without any additional descriptions or tags.
                                
                <request>
                {request}
                </request>
                """,
                input_variables=["request"]
            ).format(request=request)
      
    model_kwargs = _get_model_kwargs(temperature, top_p, top_k)  
    return BedrockClaude(
        region=config.BEDROCK_REGION,
        modelId=BedrockModel.HAIKU_3_5_CR,
        **model_kwargs
    ).invoke_llm_response(prompt)

def gen_mm_image_prompt(keyword: str,
                        image: str,
                        temperature: Optional[float] = None,
                        top_p: Optional[float] = None,
                        top_k: Optional[int] = None) -> List[str]:
    prompt = PromptTemplate(
                template=""""You are an Assistant that generates prompt for generate image by image generator model.
                The image that Human wants is written in <keyword>.
                - Write a images creation prompt keeping it to 400 characters or less.
                - If there is no <keyword>, please feel free to suggest it using your imagination.
                - If an image is given, generate a prompt that represents the keyword while maintaining the style of the given image.
                - The prompts should be clear and concise, and should be able to generate a variety of images that are relevant to the keyword.
                
                Use this fomat without further explanation:
                <prompt>image prompt</prompt>

                <keyword>
                {keyword}
                </keyword>
                """,
                input_variables=["keyword"]
            ).format(keyword=keyword)

    try:
        model_kwargs = _get_model_kwargs(temperature, top_p, top_k)
        claude = BedrockClaude(
            region=config.BEDROCK_REGION,
            modelId=BedrockModel.SONNET_3_5_CR,
            **model_kwargs
        )
        res = claude.invoke_llm_response(text=prompt, image=image)
        return _extract_format(res)[0]
    except Exception as e:
        print(e)
        return keyword

def gen_image(body: str, modelId: str):
    bedrock = boto3.client(service_name='bedrock-runtime',
                           region_name = config.BEDROCK_REGION)
    response = bedrock.invoke_model(
        body=body,
        modelId=modelId,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    image = response_body.get("images")
    return image


def create_image_params(
    seed: int,
    count: int,
    width: int,
    height: int,
    cfg: float
) -> ImageParams:
    """Create image generation parameters"""
    img_params = ImageParams(seed=seed)
    img_params.set_configuration(
        count=count,
        width=width,
        height=height,
        cfg=cfg
    )
    return img_params

def _get_model_kwargs(temperature: Optional[float] = None,
                    top_p: Optional[float] = None, 
                    top_k: Optional[int] = None) -> Dict[str, Any]:
    model_kwargs = {}
    if temperature is not None:
        model_kwargs['temperature'] = temperature
    if top_p is not None:
        model_kwargs['top_p'] = top_p
    if top_k is not None:
        model_kwargs['top_k'] = top_k
    return model_kwargs

def _extract_format(result_string):
    pattern = r'<prompt>(.*?)</prompt>'
    return re.findall(pattern, result_string)