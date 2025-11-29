from datetime import timezone
import datetime
from io import BytesIO
import boto3
from decouple import config
from botocore.client import Config

s3_client = boto3.client(
    "s3",
    endpoint_url=config("S3_CLIENT_ACCOUNT_ENDPOINT"),
    aws_access_key_id=config("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=config("R2_SECRET_ACCESS_KEY"),
    region_name="auto",
    config=Config(signature_version="s3v4")
)

def upload_video_file(filename: str, file_bytes: bytes, id_partido: str):
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    key = f"{id_partido}/{timestamp}_{filename}"
    upload(key, file_bytes)

def upload(key: str, file_bytes: bytes):
    try:
        s3_client.upload_fileobj(
            Fileobj=BytesIO(file_bytes),
            Bucket=config("R2_BUCKET"),
            Key=key,
            ExtraArgs={"ContentType": "video/mp4"}
        )

        print("Archivo subido correctamente:", key)

    except Exception as e:
        print("Error uploading to R2:", e)
        raise e
