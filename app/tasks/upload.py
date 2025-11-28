import aioboto3
from datetime import datetime, timezone
from decouple import config

async def upload_heatmap(
    filename: str,
    file_bytes: bytes,
    match_id: int,
    player_id: int
    ):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    key = f"{match_id}/{player_id}/{timestamp}_{filename}"
    await upload(key, file_bytes)
    
async def upload_player_records(
    filename: str,
    file_bytes: bytes,
    match_id: int,
    ):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    key = f"{match_id}/{timestamp}_{filename}"
    await upload(key, file_bytes)

    
async def upload(key: str, file_bytes: bytes):
    session = aioboto3.Session()
    async with session.resource('s3', region_name='auto') as s3:
        try:
            path = config("S3_API")
            await s3.Bucket(config("BUCKET")).put_object(
                Key=key,
                Body=file_bytes,
                ContentType="image/png"
            )
        except Exception as e:
            print(f"Error uploading to R2: {e}")
            raise e