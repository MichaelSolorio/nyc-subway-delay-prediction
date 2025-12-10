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

4. Route-based:
   - Route ID (L, 6, A trains)
   - Station information

WHAT IS FEATURE ENGINEERING?
Taking raw data and creating meaningful variables that help the model learn patterns.
Example: Instead of just "8:30am", we create "is_rush_hour = True"

Now supports multiple routes: L, 6, and A trains!
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import json
import os
import sys
import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config


# =============================================================================
# US FEDERAL HOLIDAYS (No API needed - hardcoded for accuracy)
# =============================================================================

def get_us_holidays(year):
    """
    Get US federal holidays for a given year.
    These are days when ridership patterns change significantly.

    Returns:
        set: Set of date objects for holidays
    """
    holidays = set()

    # Fixed-date holidays
    holidays.add(date(year, 1, 1))    # New Year's Day
    holidays.add(date(year, 7, 4))    # Independence Day
    holidays.add(date(year, 11, 11))  # Veterans Day
    holidays.add(date(year, 12, 25))  # Christmas Day

    # MLK Day: 3rd Monday of January
    holidays.add(get_nth_weekday(year, 1, 0, 3))  # Monday=0

    # Presidents Day: 3rd Monday of February
    holidays.add(get_nth_weekday(year, 2, 0, 3))

    # Memorial Day: Last Monday of May
    holidays.add(get_last_weekday(year, 5, 0))

    # Labor Day: 1st Monday of September
    holidays.add(get_nth_weekday(year, 9, 0, 1))

    # Columbus Day: 2nd Monday of October
    holidays.add(get_nth_weekday(year, 10, 0, 2))

    # Thanksgiving: 4th Thursday of November
    holidays.add(get_nth_weekday(year, 11, 3, 4))  # Thursday=3

    # Day after Thanksgiving (Black Friday) - major travel day
    thanksgiving = get_nth_weekday(year, 11, 3, 4)
    holidays.add(date(year, 11, thanksgiving.day + 1))

    return holidays


def get_nth_weekday(year, month, weekday, n):
    """Get the nth occurrence of a weekday in a month"""
    first_day = date(year, month, 1)
    first_weekday = first_day.weekday()

    # Days until first occurrence of target weekday
    days_until = (weekday - first_weekday) % 7
    first_occurrence = 1 + days_until

    # Add weeks to get to nth occurrence
    target_day = first_occurrence + (n - 1) * 7

    return date(year, month, target_day)


def get_last_weekday(year, month, weekday):
    """Get the last occurrence of a weekday in a month"""
    # Start from last day of month
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)

    last_day = next_month - pd.Timedelta(days=1)
    last_day = date(year, month, last_day.day)

    # Go back to find the weekday
    days_back = (last_day.weekday() - weekday) % 7
    return date(year, month, last_day.day - days_back)


def is_holiday(dt):
    """Check if a date is a US federal holiday"""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    d = date(dt.year, dt.month, dt.day)
    holidays = get_us_holidays(dt.year)

    return d in holidays


def is_holiday_eve(dt):
    """Check if it's the day before a major holiday (heavy travel)"""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    tomorrow = date(dt.year, dt.month, dt.day) + pd.Timedelta(days=1)

    # Handle year boundary
    if tomorrow.month == 1 and tomorrow.day == 1:
        holidays = get_us_holidays(tomorrow.year)
    else:
        holidays = get_us_holidays(dt.year)

    return tomorrow in holidays


def get_nyc_school_status(dt):
    """
    Determine if NYC public schools are likely in session.
    School schedule affects ridership significantly.

    Returns:
        dict: School-related features
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    month = dt.month
    day = dt.day

    # Summer break: roughly late June through early September
    is_summer_break = (month == 7) or (month == 8) or \
                      (month == 6 and day >= 25) or \
                      (month == 9 and day <= 7)

    # Winter break: roughly Dec 24 - Jan 2
    is_winter_break = (month == 12 and day >= 24) or \
                      (month == 1 and day <= 2)

    # Spring break: typically 3rd week of April
    is_spring_break = (month == 4 and 15 <= day <= 23)

    # School likely in session if not on break and not weekend
    is_school_day = not (is_summer_break or is_winter_break or is_spring_break) \
                    and dt.weekday() < 5  # Monday-Friday

    return {
        'is_summer_break': is_summer_break,
        'is_winter_break': is_winter_break,
        'is_spring_break': is_spring_break,
        'is_school_day': is_school_day,
    }


def get_season(month):
    """Get season from month (affects ridership and delays)"""
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'


def create_time_features(timestamp):
    """
    Create time-based features from a timestamp

    Args:
        timestamp: datetime object or ISO string

    Returns:
        dict: Time-based features including holidays, seasonality, school schedule
    """
    # Convert string to datetime if needed
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
    month = timestamp.month
    day_of_month = timestamp.day

    # Get school status
    school_status = get_nyc_school_status(timestamp)

    # Get season
    season = get_season(month)

    return {
        # Basic time features
        'hour': hour,
        'day_of_week': day_of_week,
        'day_of_month': day_of_month,
        'month': month,
        'week_of_year': timestamp.isocalendar()[1],

        # Time of day features
        'is_weekend': day_of_week >= 5,  # Saturday or Sunday
        'is_rush_hour': (7 <= hour <= 9) or (17 <= hour <= 19),  # 7-9am or 5-7pm
        'is_morning': 5 <= hour < 12,
        'is_afternoon': 12 <= hour < 17,
        'is_evening': 17 <= hour < 21,
        'is_night': hour >= 21 or hour < 5,

        # Holiday features
        'is_holiday': is_holiday(timestamp),
        'is_holiday_eve': is_holiday_eve(timestamp),

        # School schedule features
        'is_school_day': school_status['is_school_day'],
        'is_summer_break': school_status['is_summer_break'],
        'is_winter_break': school_status['is_winter_break'],

        # Season features (one-hot encoded)
        'is_winter': season == 'winter',
        'is_spring': season == 'spring',
        'is_summer': season == 'summer',
        'is_fall': season == 'fall',
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

    # Using 180 seconds (3 minutes) as threshold - this is when passengers
    # actually notice and care about delays. Also helps with class imbalance.
    DELAY_THRESHOLD = 180  # 3 minutes

    return {
        'delay_seconds': delay_seconds,
        'delay_minutes': delay_minutes,
        'is_delayed': delay_seconds > DELAY_THRESHOLD,  # More than 3 minutes late = delayed
        'is_early': delay_seconds < -DELAY_THRESHOLD,   # More than 3 minutes early
        'is_on_time': -DELAY_THRESHOLD <= delay_seconds <= DELAY_THRESHOLD,  # Within 3 minutes
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
    if delay_seconds < -180:  # More than 3 min early
        return 'early'
    elif delay_seconds <= 180:  # Within 3 minutes
        return 'on_time'
    elif delay_seconds <= 420:  # 3-7 minutes late
        return 'minor_delay'
    else:  # More than 7 minutes late
        return 'major_delay'


def load_delay_data(data_dir=None):
    """
    Load all delay JSON files from data/raw/ for all routes (L, 6, A)

    Args:
        data_dir: Directory containing JSON files

    Returns:
        pd.DataFrame: Combined delay data
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
        data_dir = os.path.abspath(data_dir)

    # Find all delay files for all routes (pattern: *_train_delays_*.json)
    pattern = os.path.join(data_dir, '*_train_delays_*.json')
    files = glob.glob(pattern)

    if not files:
        print(f"[WARNING] No delay files found in {data_dir}")
        return pd.DataFrame()

    print(f"[LOAD] Found {len(files)} delay file(s) across all routes")

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
    Load all weather feature JSON files from data/raw/ for all routes (L, 6, A)

    Args:
        data_dir: Directory containing JSON files

    Returns:
        pd.DataFrame: Combined weather data
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
        data_dir = os.path.abspath(data_dir)

    # Find all weather feature files for all routes (pattern: *_train_weather_features_*.json)
    pattern = os.path.join(data_dir, '*_train_weather_features_*.json')
    files = glob.glob(pattern)

    if not files:
        print(f"[WARNING] No weather files found in {data_dir}")
        return pd.DataFrame()

    print(f"[LOAD] Found {len(files)} weather file(s) across all routes")

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

        # Add weather columns to each row with defaults for missing values
        df['temperature'] = latest_weather.get('temperature', 50.0)  # Default ~50F
        df['humidity'] = latest_weather.get('humidity', 50.0)  # Default 50%
        df['wind_speed'] = latest_weather.get('wind_speed', 5.0)  # Default 5mph
        df['is_raining'] = latest_weather.get('is_raining', False)
        df['is_snowing'] = latest_weather.get('is_snowing', False)
        df['weather_severity'] = latest_weather.get('severity', 0)
    else:
        # No weather data - fill with reasonable defaults
        df['temperature'] = 50.0  # Moderate temperature
        df['humidity'] = 50.0  # Moderate humidity
        df['wind_speed'] = 5.0  # Light wind
        df['is_raining'] = False
        df['is_snowing'] = False
        df['weather_severity'] = 0

    # 4. Extract station info from stop_id
    print("[FEATURE] Extracting station info...")
    # Extract station number (works for L, 6, A trains: L01, 601, A01, etc.)
    df['station_number'] = df['stop_id'].str.extract(r'(\d+)')[0]
    # Fill any NaN values with 0 and convert to int
    df['station_number'] = pd.to_numeric(df['station_number'], errors='coerce').fillna(0).astype(int)

    # Extract direction (N=North/Uptown, S=South/Downtown)
    df['direction'] = df['stop_id'].str.extract(r'([NS])')[0]
    df['direction'] = df['direction'].fillna('N')  # Default to N if missing
    df['is_northbound'] = df['direction'] == 'N'

    # 5. Create the target variable for ML
    # Binary classification: is_delayed (True/False)
    # This is what we're trying to predict!
    if 'arrival_delay_seconds' in df.columns:
        df['target_is_delayed'] = df['arrival_delay_seconds'] > 180  # 3 minute threshold

    # 6. Final cleanup - fill any remaining NaN values
    print("[CLEANUP] Filling any remaining NaN values...")
    # For numeric columns, fill with 0
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # For boolean columns, fill with False
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].fillna(False)

    # Check for any remaining NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"[WARNING] Still have {nan_count} NaN values after cleanup")
    else:
        print(f"[SUCCESS] No NaN values remaining!")

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
