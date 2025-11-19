"""
Flask REST API
Serves delay predictions through HTTP endpoints

WHAT IS AN API?
Application Programming Interface - a way for programs to talk to each other
Our API lets the website (frontend) request predictions from our ML model

WHAT IS REST?
Representational State Transfer - a style of API design
Uses HTTP methods: GET (retrieve), POST (create), PUT (update), DELETE (remove)

ENDPOINTS WE'LL CREATE:
1. GET /api/health - Check if API is running
2. GET /api/predict - Get current delay predictions
3. GET /api/predict/station/<station_id> - Predictions for specific station
4. POST /api/predict/custom - Predict for custom conditions (time, weather)

HOW IT WORKS:
1. Frontend makes HTTP request to our API
2. API calls prediction model
3. Model returns prediction
4. API formats response as JSON
5. Frontend displays result to user
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from config import config
from src.models.predict import predict_current_status
from datetime import datetime

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from frontend


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns: API status and timestamp
    """
    # Simple endpoint to verify API is running
    return jsonify({
        'status': 'healthy',
        'service': 'ATLAS API',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['GET'])
def get_predictions():
    """
    Get current delay predictions for all L train stations

    Returns:
        JSON: Delay predictions with confidence scores
    """
    # TODO: Implement prediction endpoint
    # 1. Call predict_current_status()
    # 2. Format response as JSON
    # 3. Handle errors gracefully

    # Example response:
    # {
    #     'success': True,
    #     'timestamp': '2025-11-15 14:30:00',
    #     'route': 'L',
    #     'predictions': [...]
    # }

    try:
        print("🔮 API: Generating predictions...")
        # predictions = predict_current_status()
        # return jsonify(predictions)

        # Placeholder response
        return jsonify({
            'success': True,
            'message': 'Prediction endpoint (to be implemented)',
            'route': 'L'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict/station/<station_id>', methods=['GET'])
def get_station_prediction(station_id):
    """
    Get predictions for a specific station

    Args:
        station_id: Station identifier

    Returns:
        JSON: Predictions for that station
    """
    # TODO: Implement station-specific predictions
    print(f"🔮 API: Getting prediction for station {station_id}")
    return jsonify({
        'success': True,
        'station_id': station_id,
        'message': 'Station prediction (to be implemented)'
    })


@app.route('/api/stations', methods=['GET'])
def get_stations():
    """
    Get list of all L train stations

    Returns:
        JSON: List of stations
    """
    # TODO: Return list of L train stations
    # Could read from GTFS static data or hardcode
    stations = [
        {'id': 'L01', 'name': '8th Ave'},
        {'id': 'L02', 'name': '6th Ave'},
        {'id': 'L03', 'name': 'Union Square'},
        # ... add all L train stations
    ]

    return jsonify({
        'success': True,
        'route': 'L',
        'stations': stations
    })


def run_api():
    """
    Start the Flask API server
    """
    print("=" * 50)
    print("🚀 Starting ATLAS API")
    print(f"📍 Running on http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print("=" * 50)
    print()

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.DEBUG_MODE
    )


if __name__ == "__main__":
    run_api()
