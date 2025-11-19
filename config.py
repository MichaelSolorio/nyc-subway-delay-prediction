"""
ATLAS Configuration
Manages API keys, database URLs, and other settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Find the .env file in the same directory as this config.py
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

class Config:
    """Configuration class for ATLAS"""

    # MTA API Configuration
    MTA_API_KEY = os.getenv('MTA_API_KEY', '')
    MTA_GTFS_FEED_URL = 'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-l'

    # Weather API Configuration (using OpenWeatherMap as example)
    WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', '')
    WEATHER_API_URL = 'https://api.openweathermap.org/data/2.5/weather'

    # NYC Coordinates (for weather data)
    NYC_LATITUDE = 40.7128
    NYC_LONGITUDE = -74.0060

    # Database Configuration (PostgreSQL)
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/atlas_db')

    # ML Model Configuration
    MODEL_PATH = 'data/models/'
    RANDOM_FOREST_MODEL = 'random_forest_delay_model.joblib'
    XGBOOST_MODEL = 'xgboost_delay_model.joblib'

    # Data Storage Paths
    RAW_DATA_PATH = 'data/raw/'
    PROCESSED_DATA_PATH = 'data/processed/'

    # L Train Route ID (from GTFS data)
    L_TRAIN_ROUTE_ID = 'L'

    # Feature Engineering Settings
    DELAY_THRESHOLD_MINUTES = 5  # Classify as delay if >5 minutes late

    # Flask API Settings
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000
    DEBUG_MODE = os.getenv('DEBUG', 'True') == 'True'


# Create a config instance
config = Config()
