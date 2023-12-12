from typing import List
from pydantic_settings import BaseSettings,SettingsConfigDict
import os 
    
class Setting(BaseSettings):
    ROUTER_PREFIX: str
    SERVER_PORT: int
    MODEL_PATH: str
    LABEL_LIST: List[str]
    BAD_WORD_FILE_PATH: str
    PROJECT_NAME: str
    ROOT: str = os.getcwd()
    model_config = SettingsConfigDict(
        env_file= ROOT[:-3] + 'config/.env',
        extra='allow'
    )
    TOKENIZER_PATH:str

setting = Setting()