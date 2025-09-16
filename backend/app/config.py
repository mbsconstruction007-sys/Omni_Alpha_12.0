# backend/app/config.py
'''Configuration management'''

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    app_name: str = 'Omni Alpha 12.0'
    app_version: str = '12.0.0'
    debug: bool = True
    
    # API
    api_host: str = '0.0.0.0'
    api_port: int = 8000
    api_prefix: str = '/api/v1'
    
    # Database
    database_url: Optional[str] = 'sqlite:///./omni_alpha.db'
    
    # Redis
    redis_url: Optional[str] = 'redis://localhost:6379'
    
    # Trading
    trading_mode: str = 'paper'  # paper or live
    default_exchange: str = 'alpaca'
    
    # Security
    secret_key: str = 'your-secret-key-here'
    algorithm: str = 'HS256'
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = '.env'

settings = Settings()
