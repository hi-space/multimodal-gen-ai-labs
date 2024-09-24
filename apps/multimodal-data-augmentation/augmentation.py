
import re
import json
from langchain.prompts import PromptTemplate

from common.aws.claude import BedrockClaude
from common.aws.embedding import BedrockEmbedding


class Augmentation():
    def __init__(self):
        self.llm = BedrockClaude()
        self.embedding = BedrockEmbedding()


    # generate name in korean, description, tags as a JSON object
    def gen_properties(self, item: dict, image: str):
        template = PromptTemplate(
                    template="""
                        Look at the image and properties of this product and describe it in Korean following <instructions>.

                        <instructions>
                        1. Format the response as a JSON object with four keys: "category", "summary", "image_summary" and "tags".
                        - "category": Category of item
                        - "summary": Summary of product form based on appearance in a sentence
                        - "image_summary": Describe this image of product based on its type, color, material, pattern, and features.
                        - "tags": An array of strings representing key features or properties that can represent color, pattern, material, type of the product.
                        2. Review if the JSON syntax is correct.
                        3. Do not include any explanations or tags other than what was requested.
                        </instructions>
                        
                        Here are the properties of the product:
                        <product>
                        {product}
                        </product>
                        """,
                    input_variables=['product']
                )
        
        res = self.llm.invoke_llm_response(text=template.format(product=str(item)), image=image)
        try:
            j = json.loads(str(res))
            return {
                'category': j.get('category', ''),
                'summary': j.get('summary', ''),
                'image_summary': j.get('image_summary', ''),
                'tags': j.get('tags', '')
            }
        except Exception as e:
            print(e)
            return {}
    

    # generate description in Korean by json data of item
    def gen_description(self, item: dict, image: str):
        try:
            res = self.llm.invoke_llm_response(text=PromptTemplate(
                template="""
                    look at the image of the product and properties and write a detailed and narrative product description in Korean.
                    Keep a lively tone and use a hook to make users want to buy the product.
                    Do not include any explanations or tags other than what was requested.

                    Here are the properties of the product:
                    <product>
                    {product}
                    </product>
                    """,
                input_variables=['product']
            ).format(product=str(item)), image=image)
            return res
        except Exception as e:
            print(e)
            return None


    def gen_image_prompt(self, item, image: str = None):
        def _extract_format(result_string):
            pattern = r'<prompt>(.*?)</prompt>'
            return re.findall(pattern, result_string)

        text = PromptTemplate(
        template="""You are an Assistant that creates a image prompt for image generation model. Think step by step following the instructions.

            <instructions>
            1. Describe the image in as much detail as possible, referring to the <item> that contains the item's metadata.
            2. Write a prompt to generate an image in the form of a 3D icon, flat color and lighting, full shot, centered, digital, with a white background, as a design asset.
            3. The written prompt should be able to describe the given image within 400 characters. Use this fomat without further explanation:
            <prompt>image prompt</prompt>
            </instructions>

            Here are the properties of the item:
            <item>
            {item}
            </item>""",
            input_variables=['item']
        ).format(item=str(item))
       
        res = self.llm.invoke_llm_response(text=text, image=image)
        return _extract_format(res)


    # generate description about image
    def describe_image(self, image: str = None):
        res = self.llm.invoke_llm_response(
            text="Describe a item in image.",
            image=image,
        )
        return res