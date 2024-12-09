from aws_cdk import (
    Stack,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_dynamodb as dynamodb,
    aws_apprunner as apprunner,
    RemovalPolicy, 
    CfnOutput
)
from constructs import Construct


class CdkStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, app_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.app_name = app_name

        def generate_name(resource_type: str) -> str:
            return f"{self.app_name}-{resource_type}"

        # S3 버킷 생성
        bucket = s3.Bucket(
            self,
            generate_name("bucket"),
            bucket_name=generate_name("bucket-15234"),
            removal_policy=RemovalPolicy.DESTROY,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # CloudFront Distribution 생성 (OAC 설정 포함)
        distribution = cloudfront.Distribution(
            self,
            generate_name("cf"),
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3Origin(
                    bucket,
                    origin_access_identity=None  # OAI를 사용하지 않도록 설정
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            ),
        )

        # CloudFront OAC 생성
        oac = cloudfront.CfnOriginAccessControl(
            self,
            generate_name("oac"),
            origin_access_control_config=cloudfront.CfnOriginAccessControl.OriginAccessControlConfigProperty(
                name=generate_name("oac-config"),
                origin_access_control_origin_type="s3",
                signing_behavior="always",
                signing_protocol="sigv4"
            )
        )

        # OAC를 Distribution의 S3 Origin에 연결
        cfn_distribution = distribution.node.default_child
        cfn_distribution.add_property_override(
            "DistributionConfig.Origins.0.OriginAccessControlId",
            oac.attr_id
        )
        # S3 Origin의 CustomOriginConfig 제거
        cfn_distribution.add_property_override(
            "DistributionConfig.Origins.0.CustomOriginConfig",
            None
        )
       
        # S3 버킷 정책 추가
        bucket.add_to_resource_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[bucket.arn_for_objects("*")],
                principals=[iam.ServicePrincipal("cloudfront.amazonaws.com")],
                conditions={
                    "StringEquals": {
                        "AWS:SourceArn": f"arn:aws:cloudfront::{self.account}:distribution/{distribution.distribution_id}"
                    }
                },
            )
        )

        # Create DynamoDB table
        table = dynamodb.Table(
            self, self.app_name,
            table_name=self.app_name,
            partition_key=dynamodb.Attribute(
                name="id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        # 출력값 정의
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "CloudFrontDomainName", value=distribution.domain_name)
        CfnOutput(self, "TableName", value=table.table_name)