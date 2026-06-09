from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    log_level: str = "INFO"
    spacy_model: str = "en_core_web_sm"
    
    class Config:
        env_prefix = "RCT_REVIEWER_"

settings = Settings()