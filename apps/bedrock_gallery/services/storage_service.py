import boto3
import uuid
from typing import Dict, Any, BinaryIO, List
from datetime import datetime
from config import config


class StorageService:
    def __init__(self, bucket_name: str, cloudfront_domain: str):
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.bucket_name = bucket_name
        self.cloudfront_domain = cloudfront_domain
        
    def upload_image(
        self,
        image: BinaryIO,
        prompt: str,
        cfg: Dict[str, Any],
        tags: List[str]
    ) -> Dict[str, Any]:
        """Upload image to S3 and store metadata in DynamoDB"""
        # Generate unique filename
        filename = f"{uuid.uuid4()}.png"
        
        # Upload to S3
        self.s3_client.upload_fileobj(
            image,
            self.bucket_name,
            filename,
            ExtraArgs={'ContentType': 'image/png'}
        )
        
        # Store metadata in DynamoDB
        metadata = {
            'id': str(uuid.uuid4()),
            'url': f"{config.CF_DOMAIN}/{filename}",
            'prompt': prompt,
            'configuration': cfg,
            'tags': tags,
            'created_at': datetime.now().isoformat()
        }
        
        table = self.dynamodb.Table('media_metadata')
        table.put_item(Item=metadata)
        
        return metadata
    
    def get_media_list(
        self,
        media_type: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve media list from DynamoDB"""
        table = self.dynamodb.Table('media_metadata')
        
        params = {
            'Limit': limit
        }
        
        if media_type:
            params['FilterExpression'] = '#type = :type_val'
            params['ExpressionAttributeNames'] = {'#type': 'type'}
            params['ExpressionAttributeValues'] = {':type_val': media_type}
            
        response = table.scan(**params)
        return response.get('Items', [])