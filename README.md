# ATLAS - Advanced Transit Latency Analysis System

A machine learning project that predicts delays on the NYC subway system using real-time data.

## What is this?

ATLAS is a data science project I built to predict whether L trains in NYC will be delayed. It pulls live data from the MTA's public API, combines it with weather data, and uses machine learning to make predictions.

The goal was to learn how to build a full ML pipeline from scratch - from data collection all the way to a working web app.

## How it works

1. **Data Collection** - Fetches real-time L train data from MTA's GTFS feed and current weather from OpenWeatherMap
2. **Feature Engineering** - Transforms raw data into useful features (time of day, rush hour, weather conditions, etc.)
3. **Model Training** - Uses Random Forest and XGBoost to learn patterns in delay data
4. **Prediction API** - Flask REST API serves predictions to a simple web frontend

## Tech Stack

- Python (Flask, pandas, scikit-learn, XGBoost)
- MTA GTFS-realtime API
- OpenWeatherMap API
- HTML/CSS/JavaScript frontend
- Docker for deployment (planned)

## Project Structure

```
├── data/                   # Data storage (not tracked in git)
│   ├── raw/               # Raw JSON from APIs
│   ├── processed/         # Cleaned CSV files
│   └── models/            # Saved ML models
├── src/
│   ├── data_collection/   # Scripts to fetch MTA and weather data
│   ├── data_processing/   # Feature engineering pipeline
│   ├── models/            # Model training and prediction
│   └── api/               # Flask REST API
├── static/                # Frontend files
└── config.py              # Configuration and API keys
```

## Setup

1. Clone the repo
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   WEATHER_API_KEY=your_key_here
   ```
   Note: MTA data doesn't require an API key anymore.

## Running the project

Collect data:
```
venv\Scripts\python src/data_collection/mta_gtfs.py
venv\Scripts\python src/data_collection/weather_api.py
```

Process data:
```
venv\Scripts\python src/data_processing/feature_engineering.py
```

Train models (coming soon):
```
venv\Scripts\python src/models/train.py
```

Start the API:
```
venv\Scripts\python src/api/app.py
```

## Current Progress

- [x] MTA GTFS data collection
- [x] Weather API integration
- [x] Feature engineering pipeline
- [ ] Model training
- [ ] Prediction API
- [ ] Web frontend
- [ ] Docker deployment

## What I learned

- Working with real-time transit data (GTFS protocol buffers)
- API integration and handling authentication
- Feature engineering for time-series data
- Building ML pipelines from scratch
- Project organization and version control

## Author

Michael Solorio

---

Built as a portfolio project to demonstrate applied data science and ML engineering skills.
