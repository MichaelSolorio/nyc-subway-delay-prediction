"""
ATLAS - Advanced Transit Latency Analysis System
NYC Subway Delay Prediction

Main entry point for the ATLAS system
Author: Michael Solorio
"""

import sys
import argparse


def main():
    """Main entry point for ATLAS"""

    print("=" * 60)
    print("🚇 ATLAS - Advanced Transit Latency Analysis System")
    print("=" * 60)
    print()

    parser = argparse.ArgumentParser(description='ATLAS - NYC Subway Delay Prediction')

    parser.add_argument(
        'command',
        choices=['collect', 'train', 'api', 'predict'],
        help='Command to run'
    )

    args = parser.parse_args()

    if args.command == 'collect':
        print("📥 Starting data collection...")
        # TODO: Import and run data collection
        # from src.data_collection.mta_gtfs import fetch_mta_data
        # from src.data_collection.weather_api import fetch_weather_data
        print("Data collection not yet implemented")

    elif args.command == 'train':
        print("🎓 Starting model training...")
        # TODO: Import and run training
        # from src.models.train import train_all_models
        # train_all_models()
        print("Model training not yet implemented")

    elif args.command == 'api':
        print("🚀 Starting Flask API server...")
        from src.api.app import run_api
        run_api()

    elif args.command == 'predict':
        print("🔮 Making predictions...")
        # TODO: Import and run prediction
        # from src.models.predict import predict_current_status
        # results = predict_current_status()
        # print(results)
        print("Prediction not yet implemented")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print("ATLAS Commands:")
        print("  python main.py collect  - Collect MTA & weather data")
        print("  python main.py train    - Train ML models")
        print("  python main.py api      - Start Flask API server")
        print("  python main.py predict  - Make predictions")
        print()
        print("Example: python main.py api")
    else:
        main()