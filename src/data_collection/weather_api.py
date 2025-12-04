"""
Weather Data Collection
Fetches current weather data for L, 6, and A train areas to correlate with subway delays

WHY WEATHER MATTERS:
- Rain/snow can slow trains (slippery tracks, reduced visibility)
- Extreme temperatures affect tracks (heat expansion, cold contraction)
- Weather impacts ridership (more crowded = more delays)
- Storms cause service disruptions

HOW THIS WORKS:
1. We call OpenWeatherMap API for each route's area (Brooklyn, Manhattan/Bronx, Queens)
2. API returns current weather conditions
3. We extract features useful for ML prediction
4. Save data with timestamp for each location
"""

import requests
from datetime import datetime
import json
import os
import sys

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config


def fetch_weather_data(route_key):
    """
    Fetch current weather for a specific route's area from OpenWeatherMap

    Args:
        route_key: Route key ('L', '6', or 'A')

    Returns:
        dict: Raw weather data from API, or None if error
    """
    route_config = config.MTA_FEEDS.get(route_key)
    if not route_config:
        print(f"[ERROR] Unknown route: {route_key}")
        return None

    print(f"[FETCH] Fetching weather data for {route_config['name']} area...")

    # Check if API key is set
    if not config.WEATHER_API_KEY or config.WEATHER_API_KEY == 'PASTE_YOUR_WEATHER_KEY_HERE':
        print("[ERROR] WEATHER_API_KEY not set in .env file!")
        print("   Get your key at: https://openweathermap.org/api")
        return None

    # Set up the request parameters for this route's location
    params = {
        'lat': route_config['latitude'],
        'lon': route_config['longitude'],
        'appid': config.WEATHER_API_KEY,
        'units': 'imperial'  # Use Fahrenheit for temperature
    }

    try:
        # Make the HTTP request to OpenWeatherMap
        response = requests.get(config.WEATHER_API_URL, params=params)

        # Check if request was successful
        if response.status_code != 200:
            print(f"[ERROR] Weather API returned status {response.status_code}")
            if response.status_code == 401:
                print("   Your API key may be invalid")
            return None

        # Parse JSON response
        weather_data = response.json()

        # Add route identifier to the data
        weather_data['route_id'] = route_key

        print(f"[SUCCESS] Successfully fetched weather data!")
        print(f"   Location: {weather_data.get('name', 'NYC')}")
        print(f"   Temperature: {weather_data['main']['temp']}F")
        print(f"   Conditions: {weather_data['weather'][0]['description']}")

        return weather_data

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Error parsing response: {e}")
        return None


def extract_weather_features(weather_data):
    """
    Extract relevant features from weather API response for ML model

    Args:
        weather_data: Raw weather API response

    Returns:
        dict: Cleaned weather features ready for ML
    """
    if weather_data is None:
        return None

    # Get main weather condition
    main_condition = weather_data['weather'][0]['main'].lower()
    description = weather_data['weather'][0]['description'].lower()

    # Extract features
    features = {
        'timestamp': datetime.now().isoformat(),
        'route_id': weather_data.get('route_id', 'unknown'),

        # Temperature features
        'temperature': weather_data['main']['temp'],
        'feels_like': weather_data['main']['feels_like'],
        'humidity': weather_data['main']['humidity'],

        # Wind features
        'wind_speed': weather_data['wind']['speed'],

        # Condition features (booleans for ML)
        'is_raining': main_condition in ['rain', 'drizzle', 'thunderstorm'],
        'is_snowing': main_condition == 'snow',
        'is_clear': main_condition == 'clear',
        'is_cloudy': main_condition == 'clouds',

        # Raw condition for reference
        'condition': main_condition,
        'description': description,

        # Weather severity score (0-3)
        # 0 = clear, 1 = mild, 2 = moderate, 3 = severe
        'severity': calculate_severity(weather_data),

        # Visibility (in meters, if available)
        'visibility': weather_data.get('visibility', 10000),
    }

    # Add precipitation amount if raining/snowing
    if 'rain' in weather_data:
        features['rain_1h'] = weather_data['rain'].get('1h', 0)
    else:
        features['rain_1h'] = 0

    if 'snow' in weather_data:
        features['snow_1h'] = weather_data['snow'].get('1h', 0)
    else:
        features['snow_1h'] = 0

    return features


def calculate_severity(weather_data):
    """
    Calculate weather severity score for ML model

    Args:
        weather_data: Raw weather API response

    Returns:
        int: Severity score 0-3
    """
    main_condition = weather_data['weather'][0]['main'].lower()
    temp = weather_data['main']['temp']
    wind = weather_data['wind']['speed']
    visibility = weather_data.get('visibility', 10000)

    # Start with base severity from condition
    if main_condition == 'clear':
        severity = 0
    elif main_condition in ['clouds', 'mist', 'haze']:
        severity = 1
    elif main_condition in ['rain', 'drizzle']:
        severity = 2
    elif main_condition in ['thunderstorm', 'snow', 'sleet']:
        severity = 3
    else:
        severity = 1

    # Increase severity for extreme conditions
    if temp < 20 or temp > 95:  # Very cold or very hot
        severity = min(3, severity + 1)
    if wind > 25:  # High winds
        severity = min(3, severity + 1)
    if visibility < 1000:  # Low visibility
        severity = min(3, severity + 1)

    return severity


def save_weather_data(data, route_id, data_type='weather'):
    """
    Save weather data to data/raw/ folder as JSON

    Args:
        data: Weather data to save
        route_id: Route ID ('L', '6', 'A')
        data_type: Type identifier for filename
    """
    # Create data directory if it doesn't exist
    raw_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    raw_path = os.path.abspath(raw_path)
    os.makedirs(raw_path, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{route_id}_train_{data_type}_{timestamp}.json"
    filepath = os.path.join(raw_path, filename)

    # Save as JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[SAVED] Weather data to: {filename}")
    return filepath


def collect_weather_data():
    """
    Main function: Fetch, process, and save weather data for all routes

    Returns:
        dict: Summary of collected weather data
    """
    print("=" * 60)
    print("ATLAS - Weather Data Collection (L, 6, A Train Areas)")
    print("=" * 60)
    print()

    all_features = {}

    # Collect weather for each route's area
    for route_key in config.SUPPORTED_ROUTES:
        route_config = config.MTA_FEEDS[route_key]
        print(f"\n[ROUTE] Collecting weather for {route_config['name']} area")
        print("-" * 60)
        print()

        # Fetch raw weather data
        raw_data = fetch_weather_data(route_key)

        if raw_data is None:
            print(f"[ERROR] Weather data collection failed for {route_config['name']}!")
            continue

        print()

        # Extract ML-ready features
        print("[PROCESS] Extracting weather features...")
        features = extract_weather_features(raw_data)

        if features:
            print(f"   Temperature: {features['temperature']}F (feels like {features['feels_like']}F)")
            print(f"   Conditions: {features['description']}")
            print(f"   Severity: {features['severity']}/3")
            print(f"   Raining: {features['is_raining']}")
            print(f"   Snowing: {features['is_snowing']}")
            all_features[route_key] = features

        print()

        # Save both raw and processed data
        save_weather_data(raw_data, route_key, 'weather_raw')
        save_weather_data(features, route_key, 'weather_features')

    print()
    print("=" * 60)
    print("[DONE] Weather data collection complete for all routes!")
    print("=" * 60)

    return all_features


# This runs when you execute this file directly
if __name__ == "__main__":
    collect_weather_data()
