from pathlib import Path
import boto3

from app.entities.utils.singleton import Singleton
from decouple import config

class R2Downloader(metaclass=Singleton):
    def __init__(self,):
        self.s3 = boto3.client(
            "s3",
            endpoint_url=config("VIDEOS_S3_ENDPOINT"),
            aws_access_key_id=config("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=config("R2_SECRET_ACCESS_KEY"),
        )
        self.bucket = config("VIDEO_BUCKET")

    def build_destination_path(self, key: str, base_dir: str = "./tmp") -> Path:
        """
        Construye un Path válido para guardar el archivo usando pathlib.
        Extrae automáticamente el nombre del archivo desde el key.
        """
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True) 

        filename = Path(key).name
        return base / filename

    def stream_download(self, key: str, destination_path: str, chunk_size=1024*1024*16):
        """
        Descarga el archivo en chunks (16 MB por defecto).
        Soporta archivos grandes (+5GB).
        """
        with open(destination_path, "wb") as f:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"]
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                f.flush()