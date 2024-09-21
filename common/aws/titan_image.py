import boto3
import json
from botocore.config import Config


class BedrockTitanImage():
    def __init__(self, region='us-west-2', modelId = 'amazon.titan-image-generator-v2:0'):
        self.region = region
        self.modelId = modelId
        self.bedrock = boto3.client(
            service_name = 'bedrock-runtime',
            region_name = self.region,
            config = Config(
                connect_timeout=120,
                read_timeout=120,
                retries={'max_attempts': 5}
            ),
        )

    def generate_image(self, body: str):
        response = self.bedrock.invoke_model(
            body=body,
            modelId=self.modelId,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("images")