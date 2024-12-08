import boto3
import uuid
from typing import Dict, Any, BinaryIO, List, Optional
from datetime import datetime
from config import config
from genai_kit.aws.dynamodb import DynamoDB
from genai_kit.utils.random import random_id


class StorageService:
    def __init__(self, bucket_name: str, cloudfront_domain: str):
        self.s3_client = boto3.client('s3')
        self.dynamodb = DynamoDB(
            region='ap-northeast-2',
            table_name=config.DYNAMO_TABLE,
        )
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

    def upload_image(
        self,
        media_type: str,
        model_type: str,
        prompt: str,
        details: Dict[str, Any],
        status: str,
        image: BinaryIO = None,
        thumbnail: Optional[str] = None,
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        image_id = id or random_id()
        url = None

        if image:
            url = self.upload_to_s3(image, image_id)

        image_id = id or random_id()
        now = datetime.now().isoformat()
        
        url = url or f"{self.cloudfront_domain}/{image_id}"
        record = {
            "id": image_id,
            "media_type": media_type,
            "model_type": model_type,
            "prompt": prompt,
            "status": status,
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
                    'status': status,
                    'url': record['url'],
                    'thumbnail': record['thumbnail'],
                    'updated_at': now,
                    'details': details
                }
                
                self.dynamodb.update_item(image_id, updates)
                record = self.dynamodb.get_item(image_id)
            else:
                # Create new record
                record['created_at'] = now
                self.dynamodb.put_item(record)

        except Exception as e:
            raise Exception(f"Failed to store metadata in DynamoDB: {str(e)}")

        return record

    def update_status(self, id: str, status: str) -> Dict[str, Any]:
        """
        Update only the status of an existing record
        """
        try:
            self.dynamodb.update_item(id, {
                'status': status,
                'updated_at': datetime.now().isoformat()
            })
            
            return self.get_media_metadata(id)
        except Exception as e:
            raise Exception(f"Failed to update status: {str(e)}")

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