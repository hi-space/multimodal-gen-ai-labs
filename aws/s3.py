import boto3


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
        self.storage.get_object(Bucket=self.bucket_name, Key=key)
