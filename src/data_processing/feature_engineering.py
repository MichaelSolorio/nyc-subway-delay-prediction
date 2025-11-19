"""
Feature Engineering
Transforms raw data into features for machine learning

WHAT ARE FEATURES?
Features are the inputs to your ML model. Good features = better predictions!

FEATURES WE'LL CREATE:
1. Time-based:
   - Hour of day (rush hour vs off-peak)
   - Day of week (weekday vs weekend)
   - Is it rush hour? (7-9am, 5-7pm)
   - Month/season

2. Weather-based:
   - Temperature
   - Is raining/snowing?
   - Precipitation amount
   - Weather severity

3. Historical patterns:
   - Average delay for this time/day
   - Recent delay trends
   - Station-specific patterns

4. Service alerts:
   - Are there active alerts?
   - Number of alerts
   - Alert severity

WHAT IS FEATURE ENGINEERING?
Taking raw data and creating meaningful variables that help the model learn patterns.
Example: Instead of just "8:30am", we create "is_rush_hour = True"
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz


def create_time_features(timestamp):
    """
    Create time-based features from a timestamp

    Args:
        timestamp: datetime object

    Returns:
        dict: Time-based features
    """
    # TODO: Implement time feature creation
    # Convert timestamp to:
    # - hour (0-23)
    # - day_of_week (0-6)
    # - is_rush_hour (boolean)
    # - is_weekend (boolean)
    # - month, season, etc.

    pass


def create_weather_features(weather_data):
    """
    Create weather-based features

    Args:
        weather_data: Dict with weather information

    Returns:
        dict: Weather features for ML model
    """
    # TODO: Transform weather data into features
    # - Normalize temperature
    # - Create categorical variables (rain/snow/clear)
    # - Weather severity score
    pass


def create_historical_features(current_time, station_id, historical_data):
    """
    Create features based on historical patterns

    Args:
        current_time: Current timestamp
        station_id: Station identifier
        historical_data: DataFrame with past delay data

    Returns:
        dict: Historical pattern features
    """
    # TODO: Calculate historical averages
    # - Average delay for this hour/day combination
    # - Recent trend (last week's average)
    # - Station-specific patterns
    pass


def engineer_all_features(mta_data, weather_data, historical_data=None):
    """
    Master function: combines all feature engineering

    Args:
        mta_data: Raw MTA data
        weather_data: Raw weather data
        historical_data: Historical delay data (optional)

    Returns:
        pd.DataFrame: Complete feature set ready for ML model
    """
    # TODO: Combine all feature engineering steps
    # 1. Extract time features
    # 2. Add weather features
    # 3. Add historical features (if available)
    # 4. Return complete feature DataFrame
    pass


if __name__ == "__main__":
    print("Feature engineering module")
    print("This transforms raw data into ML-ready features")
