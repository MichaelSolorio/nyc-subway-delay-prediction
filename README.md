# 🚇 ATLAS - Advanced Transit Latency Analysis System

A machine learning-powered web application designed to predict delays in the New York City subway system with real-time accuracy.

## 📋 Project Overview

ATLAS leverages publicly available MTA GTFS-realtime data, weather APIs, and historical transit patterns to predict whether trains will arrive on-time or experience delays, along with estimated delay severity.

**Current Focus:** L train route

## 🎯 Features

- Real-time delay prediction using MTA GTFS data
- Weather-aware predictions
- Historical pattern analysis
- REST API for predictions
- Interactive web interface
- Focus on L train route (expandable to other routes)

## 🛠️ Technology Stack

- **Backend:** Python, Flask
- **Machine Learning:** Random Forest, XGBoost (scikit-learn)
- **Data Sources:** MTA GTFS-realtime API, Weather API
- **Database:** PostgreSQL
- **Deployment:** Docker

## 📁 Project Structure

```
├── data/                   # Data storage (git-ignored)
│   ├── raw/               # Raw downloaded data
│   ├── processed/         # Cleaned/processed data
│   └── models/            # Saved ML models
├── src/                   # Source code
│   ├── data_collection/   # MTA & weather data fetching
│   ├── data_processing/   # Feature engineering & cleaning
│   ├── models/            # ML model training & prediction
│   └── api/               # Flask REST API
├── static/                # Frontend (HTML/CSS/JS)
├── notebooks/             # Jupyter notebooks for exploration
└── docker/                # Docker configuration
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip
- PostgreSQL (optional for now)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables (create `.env` file):
   ```
   MTA_API_KEY=your_mta_api_key
   WEATHER_API_KEY=your_weather_api_key
   ```

## 📊 Development Roadmap

- [ ] Phase 1: Data Collection Setup
  - [ ] MTA GTFS-realtime data fetching
  - [ ] Weather API integration
- [ ] Phase 2: Data Processing
  - [ ] Feature engineering
  - [ ] Data cleaning pipeline
- [ ] Phase 3: Model Development
  - [ ] Train Random Forest model
  - [ ] Train XGBoost model
  - [ ] Model evaluation
- [ ] Phase 4: API Development
  - [ ] Flask REST API
  - [ ] Prediction endpoints
- [ ] Phase 5: Frontend
  - [ ] Web interface
  - [ ] Real-time predictions display
- [ ] Phase 6: Deployment
  - [ ] Docker containerization
  - [ ] PostgreSQL integration

## 👨‍💻 Author

Michael Solorio

## 📝 License

This project is for educational and portfolio purposes.
