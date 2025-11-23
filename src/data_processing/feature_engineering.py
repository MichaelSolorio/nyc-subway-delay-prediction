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

2. Weather-based:
   - Temperature
   - Is raining/snowing?
   - Weather severity

3. Delay-based:
   - Delay in seconds
   - Is delayed? (yes/no)
   - Delay category (on-time, minor, major)

WHAT IS FEATURE ENGINEERING?
Taking raw data and creating meaningful variables that help the model learn patterns.
Example: Instead of just "8:30am", we create "is_rush_hour = True"
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys
import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config


def create_time_features(timestamp):
    """
    Create time-based features from a timestamp

    Args:
        timestamp: datetime object or ISO string

    Returns:
        dict: Time-based features
    """
    # Convert string to datetime if needed
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday

    return {
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': day_of_week >= 5,  # Saturday or Sunday
        'is_rush_hour': (7 <= hour <= 9) or (17 <= hour <= 19),  # 7-9am or 5-7pm
        'is_morning': 5 <= hour < 12,
        'is_afternoon': 12 <= hour < 17,
        'is_evening': 17 <= hour < 21,
        'is_night': hour >= 21 or hour < 5,
    }


def create_delay_features(delay_seconds):
    """
    Create delay-based features from raw delay value

    Args:
        delay_seconds: Delay in seconds (negative = early, positive = late)

    Returns:
        dict: Delay features
    """
    delay_minutes = delay_seconds / 60

    return {
        'delay_seconds': delay_seconds,
        'delay_minutes': delay_minutes,
        'is_delayed': delay_seconds > 60,  # More than 1 minute late = delayed
        'is_early': delay_seconds < -60,   # More than 1 minute early
        'is_on_time': -60 <= delay_seconds <= 60,  # Within 1 minute
        'delay_category': categorize_delay(delay_seconds),
    }


def categorize_delay(delay_seconds):
    """
    Categorize delay severity

    Args:
        delay_seconds: Delay in seconds

    Returns:
        str: Category ('early', 'on_time', 'minor_delay', 'major_delay')
    """
    if delay_seconds < -60:
        return 'early'
    elif delay_seconds <= 60:
        return 'on_time'
    elif delay_seconds <= 300:  # Up to 5 minutes
        return 'minor_delay'
    else:
        return 'major_delay'


def load_delay_data(data_dir=None):
    """
    Load all delay JSON files from data/raw/

    Args:
        data_dir: Directory containing JSON files

    Returns:
        pd.DataFrame: Combined delay data
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
        data_dir = os.path.abspath(data_dir)

    # Find all delay files
    pattern = os.path.join(data_dir, 'l_train_delays_*.json')
    files = glob.glob(pattern)

    if not files:
        print(f"[WARNING] No delay files found in {data_dir}")
        return pd.DataFrame()

    print(f"[LOAD] Found {len(files)} delay file(s)")

    # Load and combine all files
    all_data = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.extend(data)

    print(f"[LOAD] Loaded {len(all_data)} total delay records")
    return pd.DataFrame(all_data)


def load_weather_data(data_dir=None):
    """
    Load all weather feature JSON files from data/raw/

    Args:
        data_dir: Directory containing JSON files

    Returns:
        pd.DataFrame: Combined weather data
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
        data_dir = os.path.abspath(data_dir)

    # Find all weather feature files
    pattern = os.path.join(data_dir, 'nyc_weather_features_*.json')
    files = glob.glob(pattern)

    if not files:
        print(f"[WARNING] No weather files found in {data_dir}")
        return pd.DataFrame()

    print(f"[LOAD] Found {len(files)} weather file(s)")

    # Load and combine all files
    all_data = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.append(data)

    print(f"[LOAD] Loaded {len(all_data)} weather records")
    return pd.DataFrame(all_data)


def engineer_features(delay_df, weather_df):
    """
    Main function: Engineer all features for ML

    Args:
        delay_df: DataFrame with delay data
        weather_df: DataFrame with weather data

    Returns:
        pd.DataFrame: Complete feature set ready for ML
    """
    if delay_df.empty:
        print("[ERROR] No delay data to process")
        return pd.DataFrame()

    print(f"[PROCESS] Engineering features for {len(delay_df)} records...")

    # Create a copy to avoid modifying original
    df = delay_df.copy()

    # 1. Create time features from timestamp
    print("[FEATURE] Creating time features...")
    time_features = df['timestamp'].apply(create_time_features)
    time_df = pd.DataFrame(time_features.tolist())
    df = pd.concat([df, time_df], axis=1)

    # 2. Create delay features
    print("[FEATURE] Creating delay features...")
    # Use arrival_delay_seconds if available
    if 'arrival_delay_seconds' in df.columns:
        delay_features = df['arrival_delay_seconds'].apply(create_delay_features)
        delay_df_features = pd.DataFrame(delay_features.tolist())
        df = pd.concat([df, delay_df_features], axis=1)

    # 3. Add weather features (use most recent weather for all records)
    print("[FEATURE] Adding weather features...")
    if not weather_df.empty:
        # Get the most recent weather data
        latest_weather = weather_df.iloc[-1]

        # Add weather columns to each row
        df['temperature'] = latest_weather.get('temperature', None)
        df['humidity'] = latest_weather.get('humidity', None)
        df['wind_speed'] = latest_weather.get('wind_speed', None)
        df['is_raining'] = latest_weather.get('is_raining', False)
        df['is_snowing'] = latest_weather.get('is_snowing', False)
        df['weather_severity'] = latest_weather.get('severity', 0)
    else:
        # No weather data - fill with defaults
        df['temperature'] = None
        df['humidity'] = None
        df['wind_speed'] = None
        df['is_raining'] = False
        df['is_snowing'] = False
        df['weather_severity'] = 0

    # 4. Extract station info from stop_id
    print("[FEATURE] Extracting station info...")
    df['station_number'] = df['stop_id'].str.extract(r'L(\d+)')[0].astype(int)
    df['direction'] = df['stop_id'].str.extract(r'L\d+([NS])')[0]
    df['is_northbound'] = df['direction'] == 'N'

    # 5. Create the target variable for ML
    # Binary classification: is_delayed (True/False)
    # This is what we're trying to predict!
    if 'arrival_delay_seconds' in df.columns:
        df['target_is_delayed'] = df['arrival_delay_seconds'] > 60

    print(f"[DONE] Created {len(df.columns)} features")

    return df


def save_processed_data(df, filename='processed_features.csv'):
    """
    Save processed data to data/processed/

    Args:
        df: DataFrame to save
        filename: Output filename
    """
    processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    processed_dir = os.path.abspath(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    filepath = os.path.join(processed_dir, filename)
    df.to_csv(filepath, index=False)

    print(f"[SAVED] Processed data to: {filename}")
    return filepath


def process_all_data():
    """
    Main pipeline: Load, process, and save all data
    """
    print("=" * 50)
    print("ATLAS - Feature Engineering Pipeline")
    print("=" * 50)
    print()

    # Load data
    delay_df = load_delay_data()
    weather_df = load_weather_data()

    if delay_df.empty:
        print("[ERROR] No data to process!")
        return None

    print()

    # Engineer features
    processed_df = engineer_features(delay_df, weather_df)

    print()

    # Show summary
    print("[SUMMARY] Feature Engineering Results:")
    print(f"   Total records: {len(processed_df)}")
    print(f"   Total features: {len(processed_df.columns)}")

    if 'target_is_delayed' in processed_df.columns:
        delayed_count = processed_df['target_is_delayed'].sum()
        delayed_pct = (delayed_count / len(processed_df)) * 100
        print(f"   Delayed trains: {delayed_count} ({delayed_pct:.1f}%)")
        print(f"   On-time trains: {len(processed_df) - delayed_count} ({100-delayed_pct:.1f}%)")

    print()

    # Save processed data
    save_processed_data(processed_df)

    print()
    print("[DONE] Feature engineering complete!")

    return processed_df


if __name__ == "__main__":
    process_all_data()
