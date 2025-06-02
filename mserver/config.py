from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gcs_bucket: str = "qianminj-bucket"
    model_file: str = "exported_tpu_linear1015"
    weight_file: str = "tmp/dataset3"

settings = Settings()
