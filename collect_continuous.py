"""
Continuous Data Collection
Automatically collects MTA and weather data for L, 6, and A trains at regular intervals

HOW TO USE:
1. Run this script: venv\Scripts\python collect_continuous.py
2. Leave it running in the background
3. Press Ctrl+C to stop

It will collect data every 10 minutes and save to data/raw/
Collects subway data and weather for all 3 routes: L, 6, and A trains
The more varied data you collect (different times, days, weather),
the better your ML model will be!
"""

import time
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_collection.mta_gtfs import collect_all_data as collect_mta
from src.data_collection.weather_api import collect_weather_data as collect_weather


# How often to collect (in seconds)
COLLECTION_INTERVAL = 600  # 10 minutes = 600 seconds


def run_collection_cycle():
    """Run one collection cycle for MTA and weather data"""
    print()
    print("=" * 60)
    print(f"COLLECTION CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Collect MTA data
    try:
        mta_result = collect_mta()
        if mta_result:
            total_delays = sum(route['delays'] for route in mta_result.values())
            print(f"\n[MTA] Collected {total_delays} total delay records across all routes")
    except Exception as e:
        print(f"[ERROR] MTA collection failed: {e}")

    print()

    # Collect weather data
    try:
        weather_result = collect_weather()
        if weather_result:
            # Show average temperature across all routes
            temps = [route['temperature'] for route in weather_result.values()]
            avg_temp = sum(temps) / len(temps)
            print(f"\n[WEATHER] Average temperature: {avg_temp:.1f}F")
    except Exception as e:
        print(f"[ERROR] Weather collection failed: {e}")

    print()
    print("-" * 60)


def main():
    """Main loop - continuously collect data"""
    print()
    print("=" * 60)
    print("ATLAS - Continuous Data Collection (L, 6, A Trains)")
    print("=" * 60)
    print()
    print(f"Collecting data every {COLLECTION_INTERVAL // 60} minutes")
    print("Routes: L Train, 6 Train, A Train")
    print("Press Ctrl+C to stop")
    print()
    print("-" * 60)

    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            print(f"\n[CYCLE {cycle_count}]")

            # Run collection
            run_collection_cycle()

            # Calculate next run time
            next_run = datetime.now().timestamp() + COLLECTION_INTERVAL
            next_run_str = datetime.fromtimestamp(next_run).strftime('%H:%M:%S')

            print(f"[WAITING] Next collection at {next_run_str}")
            print(f"          (or press Ctrl+C to stop)")

            # Wait for next cycle
            time.sleep(COLLECTION_INTERVAL)

    except KeyboardInterrupt:
        print()
        print()
        print("=" * 60)
        print("COLLECTION STOPPED")
        print("=" * 60)
        print(f"Completed {cycle_count} collection cycles")
        print()
        print("Your data is saved in data/raw/")
        print("Run feature engineering and model training to use it:")
        print("  venv\\Scripts\\python src/data_processing/feature_engineering.py")
        print("  venv\\Scripts\\python src/models/train.py")
        print()


if __name__ == "__main__":
    main()
