"""
Configuration and settings for NutriGraph application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # Backend API configuration
    BACKEND_URL: str = os.getenv("NUTRIGRAPH_BACKEND_URL", "http://localhost:8000")
    ENVIRONMENT: str = os.getenv("NUTRIGRAPH_ENV", "local")
    
    # Application metadata
    APP_TITLE: str = "NutriGraph"
    APP_ICON: str = "🥗"
    
    # Environment options
    ENVIRONMENTS: list[str] = ["Local", "Staging"]
    
    # Default values for mock data
    DEFAULT_SERVING_SIZE: str = "1 serving"
    DEFAULT_UNITS: list[str] = ["g", "oz", "cup", "tbsp", "tsp", "piece", "ml"]


# Singleton instance
settings = Settings()
