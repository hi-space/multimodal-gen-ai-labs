import boto3
import uuid
from typing import Dict, Any, BinaryIO, List, Optional
from datetime import datetime
from config import config
from genai_kit.aws.bedrock import BedrockModel
from genai_kit.aws.dynamodb import DynamoDB
from genai_kit.utils.random import random_id
from apps.bedrock_gallery.utils import extract_key_from_uri
from apps.bedrock_gallery.types import MediaType


class StorageService:
    def __init__(self, bucket_name: str, cloudfront_domain: str):
        self.s3_client = boto3.client('s3')
        self.dynamodb = DynamoDB(table_name=config.DYNAMO_TABLE)
        self.bucket_name = bucket_name
        self.cloudfront_domain = cloudfront_domain
        
    def upload_to_s3(self, image: BinaryIO, image_id: str) -> str:
        filename = f"{image_id}.png"
        
        try:
            self.s3_client.upload_fileobj(
                image,
                self.bucket_name,
                filename,
                ExtraArgs={'ContentType': 'image/png'}
            )
            return f"{self.cloudfront_domain}/{filename}"
        except Exception as e:
            raise Exception(f"Failed to upload image to S3: {str(e)}")
        
    def upload_media(
        self,
        media_type: MediaType,
        model_type: BedrockModel,
        prompt: str,
        details: Optional[Dict[str, Any]] = None,
        image: Optional[BinaryIO] = None,
    ):
        if media_type == MediaType.IMAGE:
            return self.upload_image(
                model_type=model_type.value,
                prompt=prompt,
                details=details,
                image=image,
            )
        elif media_type == MediaType.VIDEO:
            s3Uri = details.get("outputDataConfig", {}).get("s3OutputDataConfig", {}).get("s3Uri", "")
            id = extract_key_from_uri(s3Uri)
            return self.update_video_status(
                model_type=model_type.value,
                prompt=prompt,
                details=details,
                image=image,
                id=id,
            )    

    def upload_image(
        self,
        model_type: str,
        prompt: str,
        details: Dict[str, Any],
        image: Optional[BinaryIO] = None,
        thumbnail: Optional[str] = None,
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        image_id = id or random_id()
        now = datetime.now().isoformat()
        url = None

        if image:
            url = self.upload_to_s3(image, image_id)

        url = url or f"{self.cloudfront_domain}/{image_id}"
        record = {
            "id": image_id,
            "media_type": MediaType.IMAGE.value,
            "model_type": model_type,
            "prompt": prompt,
            "url": url,
            "thumbnail": thumbnail or url,
            "updated_at": now,
            "details": details
        }

        try:
            existing_item = self.dynamodb.get_item(image_id)
            
            if existing_item:
                updates = {
                    'model_type': model_type,
                    'url': record['url'],
                    'thumbnail': record['thumbnail'],
                    'updated_at': now,
                    'details': details
                }
                
                self.dynamodb.update_item(image_id, updates)
                record = self.dynamodb.get_item(image_id)
            else:
                record['created_at'] = now
                self.dynamodb.put_item(record)

        except Exception as e:
            raise Exception(f"Failed to store metadata in DynamoDB: {str(e)}")

        return record
    
    def update_video_status(
        self,
        model_type: str,
        prompt: str,
        details: Dict[str, Any],
        image: Optional[BinaryIO] = None,
        thumbnail: Optional[str] = None,
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        id = id or random_id()
        now = datetime.now().isoformat()
        
        url = ''
        if image:
            url = self.upload_to_s3(image, id)

        record = {
            "id": id,
            "media_type": MediaType.VIDEO.value,
            "model_type": model_type,
            "prompt": prompt,
            "url": url,
            "thumbnail": thumbnail or url,
            "updated_at": now,
            "details": details
        }

        try:
            existing_item = self.dynamodb.get_item(id)
            
            if existing_item:
                updates = {
                    'url': record['url'],
                    'thumbnail': record['thumbnail'],
                    'updated_at': now,
                    'details': details
                }
                
                self.dynamodb.update_item(id, updates)
                record = self.dynamodb.get_item(id)
            else:
                record['created_at'] = now
                self.dynamodb.put_item(record)

        except Exception as e:
            raise Exception(f"Failed to store metadata in DynamoDB: {str(e)}")

        print('record', record)
        return record

    def get_media_metadata(self, id: str) -> Dict[str, Any]:
        """
        Retrieve media metadata from DynamoDB by ID
        """
        try:
            response = self.dynamodb.get_item(id)
            if not response:
                return None
            return response
        except Exception as e:
            raise Exception(f"Failed to retrieve metadata: {str(e)}")

    def get_media_list(
        self,
        media_type: str = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve media list from DynamoDB"""
        try:
            if media_type:
                query = {
                    'FilterExpression': '#type = :type_val',
                    'ExpressionAttributeNames': {'#type': 'media_type'},
                    'ExpressionAttributeValues': {':type_val': media_type}
                }
                response = self.dynamodb.scan_items(query)
            else:
                response = self.dynamodb.scan_items({})
                
            return response.get('Items', [])
        except Exception as e:
            raise Exception(f"Failed to retrieve media list: {str(e)}")