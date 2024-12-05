import boto3
from urllib.parse import urlparse


class S3:
    def __init__(self, bucket_name, region='us-west-2'):
        self.storage = boto3.client('s3',                            
                                    region_name = region)
        self.bucket_name = bucket_name

    def upload_object(self, bytes, key, extra_args={}):
        self.storage.upload_fileobj(
            bytes,
            self.bucket_name,
            key,
            ExtraArgs=extra_args
        )

    def get_object(self, key):
        response = self.storage.get_object(Bucket=self.bucket_name, Key=key)
        content = response['Body'].read()
        return content

    def download_object(self, key, file_path):
        self.storage.download_file(self.bucket_name, key, file_path)

    def extract_key_from_uri(self, s3_uri):
        parsed_uri = urlparse(s3_uri)
        return parsed_uri.path.lstrip('/')