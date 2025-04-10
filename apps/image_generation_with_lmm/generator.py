import boto3
import re
import json

from typing import List, Optional
from genai_kit.aws.claude import BedrockClaude
from genai_kit.aws.amazon_image import BedrockAmazonImage
from genai_kit.aws.amazon_video import BedrockAmazonVideo
from prompt import (
    get_llm_image_prompt,
    get_mm_llm_image_prompt,
    get_image_tags_prompt,
    get_translate_llm_prompt,
)


def gen_english(request: str):
    prompt = get_translate_llm_prompt(request=request)
    return BedrockClaude().invoke_llm_response(prompt)

def gen_image_prompt(request: str,
                     style: str,
                     temperature: Optional[float] = None,
                     top_p: Optional[float] = None,
                     top_k: Optional[int] = None) -> List[str]:
    prompt = get_llm_image_prompt(request=request, style=style)

    model_kwargs = {}
    if temperature is not None:
        model_kwargs['temperature'] = temperature
    if top_p is not None:
        model_kwargs['top_p'] = top_p
    if top_k is not None:
        model_kwargs['top_k'] = top_k

    claude = BedrockClaude(**model_kwargs)
    res = claude.invoke_llm_response(prompt)
    return _extract_format(res)


def gen_mm_image_prompt(request: str,
                        image: str,
                        temperature: Optional[float] = None,
                        top_p: Optional[float] = None,
                        top_k: Optional[int] = None) -> List[str]:
    prompt = get_mm_llm_image_prompt(request=request)

    model_kwargs = {}
    if temperature is not None:
        model_kwargs['temperature'] = temperature
    if top_p is not None:
        model_kwargs['top_p'] = top_p
    if top_k is not None:
        model_kwargs['top_k'] = top_k

    claude = BedrockClaude(**model_kwargs)
    res = claude.invoke_llm_response(text=prompt, image=image)
    return _extract_format(res)


def gen_image(body: str):
    bedrock = boto3.client(service_name='bedrock-runtime',
                           region_name = 'us-west-2')
    response = bedrock.invoke_model(
        body=body,
        modelId="amazon.titan-image-generator-v2:0",
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    image = response_body.get("images")
    return image


def gen_tags(image: str):
    prompt = get_image_tags_prompt()
    claude = BedrockClaude()
    res = claude.invoke_llm_response(text=prompt, image=image)
    return res


def _extract_format(result_string):
    pattern = r'<prompt>(.*?)</prompt>'
    return re.findall(pattern, result_string)