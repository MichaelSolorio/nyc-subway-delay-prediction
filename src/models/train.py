"""
Model Training
Trains Random Forest and XGBoost models to predict delays

MACHINE LEARNING BASICS:
1. SUPERVISED LEARNING: We have examples with answers (historical data with delays)
2. CLASSIFICATION: Predict categories (on-time vs delayed)

THE PROCESS:
1. Load cleaned, feature-engineered data
2. Split into training set (80%) and test set (20%)
3. Train models on training set
4. Evaluate on test set
5. Save best model

MODELS IN USE:
- Random Forest: Ensemble of decision trees (robust, good baseline)
- XGBoost: Gradient boosting (usually more accurate, industry standard)
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config


# Features we'll use for prediction
# These are the columns that the model will learn from
FEATURE_COLUMNS = [
    # Basic time features
    'hour',
    'day_of_week',
    'day_of_month',
    'month',
    'week_of_year',

    # Time of day features
    'is_weekend',
    'is_rush_hour',
    'is_morning',
    'is_evening',
    'is_night',

    # Holiday & calendar features (NEW)
    'is_holiday',
    'is_holiday_eve',
    'is_school_day',
    'is_summer_break',
    'is_winter_break',

    # Season features (NEW)
    'is_winter',
    'is_spring',
    'is_summer',
    'is_fall',

    # Weather features
    'temperature',
    'humidity',
    'wind_speed',
    'is_raining',
    'is_snowing',
    'weather_severity',

    # Station features
    'station_number',
    'is_northbound',
]

# What we're predicting
TARGET_COLUMN = 'target_is_delayed'


def balance_classes(X, y, target_ratio=0.35):
    """
    Balance classes using smart undersampling

    Keeps ALL delayed samples (minority class) and randomly samples
    on-time trains to achieve target ratio of delayed samples.

    Args:
        X: Features DataFrame
        y: Target Series
        target_ratio: Desired ratio of delayed samples (default 35%)

    Returns:
        tuple: Balanced (X, y)
    """
    print("[BALANCE] Balancing classes with smart undersampling...")

    # Separate delayed and on-time samples
    delayed_mask = y == True
    ontime_mask = y == False

    X_delayed = X[delayed_mask]
    y_delayed = y[delayed_mask]
    X_ontime = X[ontime_mask]
    y_ontime = y[ontime_mask]

    n_delayed = len(X_delayed)
    n_ontime = len(X_ontime)

    print(f"   Original: {n_delayed} delayed, {n_ontime} on-time")
    print(f"   Original ratio: {n_delayed/(n_delayed+n_ontime)*100:.1f}% delayed")

    # Calculate how many on-time samples we need for target ratio
    # target_ratio = n_delayed / (n_delayed + n_ontime_sampled)
    # Solving: n_ontime_sampled = n_delayed * (1 - target_ratio) / target_ratio
    n_ontime_target = int(n_delayed * (1 - target_ratio) / target_ratio)

    # Don't sample more than we have
    n_ontime_sample = min(n_ontime_target, n_ontime)

    # Randomly sample on-time trains
    np.random.seed(42)  # For reproducibility
    ontime_indices = np.random.choice(X_ontime.index, size=n_ontime_sample, replace=False)

    X_ontime_sampled = X_ontime.loc[ontime_indices]
    y_ontime_sampled = y_ontime.loc[ontime_indices]

    # Combine
    X_balanced = pd.concat([X_delayed, X_ontime_sampled])
    y_balanced = pd.concat([y_delayed, y_ontime_sampled])

    # Shuffle
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced.iloc[shuffle_idx].reset_index(drop=True)
    y_balanced = y_balanced.iloc[shuffle_idx].reset_index(drop=True)

    print(f"   Balanced: {len(y_balanced[y_balanced==True])} delayed, {len(y_balanced[y_balanced==False])} on-time")
    print(f"   New ratio: {y_balanced.mean()*100:.1f}% delayed")

    return X_balanced, y_balanced


def load_training_data():
    """
    Load processed data for training

    Returns:
        tuple: (X, y) features and labels, or (None, None) if error
    """
    print("[LOAD] Loading training data...")

    # Find the processed data file
    processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    processed_dir = os.path.abspath(processed_dir)
    filepath = os.path.join(processed_dir, 'processed_features.csv')

    if not os.path.exists(filepath):
        print(f"[ERROR] No processed data found at {filepath}")
        print("   Run feature_engineering.py first!")
        return None, None

    # Load the CSV
    df = pd.read_csv(filepath)
    print(f"[LOAD] Loaded {len(df)} records with {len(df.columns)} columns")

    # Check if we have the target column
    if TARGET_COLUMN not in df.columns:
        print(f"[ERROR] Target column '{TARGET_COLUMN}' not found!")
        return None, None

    # Select only the features we want
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing_features:
        print(f"[WARNING] Missing features: {missing_features}")

    print(f"[LOAD] Using {len(available_features)} features")

    # Get features (X) and target (y)
    X = df[available_features].copy()
    y = df[TARGET_COLUMN].copy()

    # Handle missing values (fill with median for numeric, False for boolean)
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(False)

    # Convert boolean columns to int (0/1) for sklearn
    bool_columns = X.select_dtypes(include=['bool']).columns
    X[bool_columns] = X[bool_columns].astype(int)

    print(f"[LOAD] Features shape: {X.shape}")
    print(f"[LOAD] Target distribution: {y.value_counts().to_dict()}")

    # Balance classes using smart undersampling
    X, y = balance_classes(X, y, target_ratio=0.35)

    return X, y


def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets

    Args:
        X: Features
        y: Labels
        test_size: Fraction for test set (default 20%)

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"[SPLIT] Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,  # For reproducibility
        stratify=y  # Keep same ratio of delayed/on-time in both sets
    )

    print(f"[SPLIT] Training set: {len(X_train)} samples")
    print(f"[SPLIT] Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier

    Random Forest = Many decision trees working together
    - Each tree sees a random subset of data
    - Final prediction = majority vote of all trees
    - Robust and hard to overfit

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        RandomForestClassifier: Trained model
    """
    print("[TRAIN] Training Random Forest model...")
    print("   Using class_weight='balanced' to handle class imbalance")

    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees in the forest
        max_depth=10,          # How deep each tree can grow
        min_samples_split=5,   # Minimum samples needed to split a node
        random_state=42,       # For reproducibility
        n_jobs=-1,             # Use all CPU cores
        class_weight='balanced'  # CRITICAL: Give more weight to minority class (delays)
    )

    model.fit(X_train, y_train)

    print("[TRAIN] Random Forest training complete!")
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier

    XGBoost = Trees that learn from each other's mistakes
    - Each new tree tries to fix errors from previous trees
    - Very powerful, often wins ML competitions
    - Industry standard for tabular data

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        XGBClassifier: Trained model
    """
    print("[TRAIN] Training XGBoost model...")

    # Calculate scale_pos_weight to handle class imbalance
    # This tells XGBoost how much more to weight the minority class
    num_negative = (y_train == 0).sum()  # On-time trains
    num_positive = (y_train == 1).sum()  # Delayed trains
    scale_pos_weight = num_negative / num_positive

    print(f"   Class imbalance: {num_negative} on-time, {num_positive} delayed")
    print(f"   Using scale_pos_weight={scale_pos_weight:.2f} to balance classes")

    model = xgb.XGBClassifier(
        n_estimators=100,      # Number of boosting rounds
        max_depth=6,           # Tree depth
        learning_rate=0.1,     # How fast it learns (smaller = more careful)
        random_state=42,
        eval_metric='logloss', # Metric to optimize
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight  # CRITICAL: Balance the classes
    )

    model.fit(X_train, y_train)

    print("[TRAIN] XGBoost training complete!")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance

    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        model_name: Name for display

    Returns:
        dict: Evaluation metrics
    """
    print(f"[EVAL] Evaluating {model_name}...")

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)

    # Confusion matrix: [[true_neg, false_pos], [false_neg, true_pos]]
    cm = confusion_matrix(y_test, predictions)

    # Classification report (precision, recall, f1)
    report = classification_report(y_test, predictions, output_dict=True)

    print(f"\n{'='*40}")
    print(f"{model_name} Results:")
    print(f"{'='*40}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 On-Time  Delayed")
    print(f"Actual On-Time    {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Actual Delayed    {cm[1][0]:4d}    {cm[1][1]:4d}")

    if 'True' in report or True in report:
        key = True if True in report else 'True'
        print(f"\nFor Delayed trains:")
        print(f"  Precision: {report[key]['precision']:.1%} (when we predict delay, how often correct)")
        print(f"  Recall: {report[key]['recall']:.1%} (of actual delays, how many we caught)")
        print(f"  F1-Score: {report[key]['f1-score']:.1%} (balance of precision and recall)")

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'report': report
    }


def get_feature_importance(model, feature_names, model_name="Model"):
    """
    Show which features are most important for predictions

    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name for display
    """
    print(f"\n[IMPORTANCE] {model_name} Feature Importance:")

    # Get importance scores
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    print("   Rank | Feature              | Importance")
    print("   " + "-"*45)
    for i, idx in enumerate(indices[:10]):  # Top 10
        print(f"   {i+1:4d} | {feature_names[idx]:20s} | {importances[idx]:.4f}")


def save_model(model, model_name):
    """
    Save trained model to disk

    Args:
        model: Trained model
        model_name: Filename for saved model
    """
    # Create models directory if needed
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models')
    model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    filepath = os.path.join(model_dir, model_name)
    joblib.dump(model, filepath)

    print(f"[SAVED] Model saved to: {model_name}")
    return filepath


def train_all_models():
    """
    Master training function
    Trains both Random Forest and XGBoost, compares them
    """
    print("=" * 50)
    print("ATLAS - Model Training Pipeline")
    print("=" * 50)
    print()

    # 1. Load data
    X, y = load_training_data()
    if X is None:
        return None

    print()

    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    print()

    # 3. Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    get_feature_importance(rf_model, X.columns.tolist(), "Random Forest")

    print()

    # 4. Train XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    get_feature_importance(xgb_model, X.columns.tolist(), "XGBoost")

    print()

    # 5. Compare and save
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Random Forest Accuracy: {rf_results['accuracy']:.1%}")
    print(f"XGBoost Accuracy: {xgb_results['accuracy']:.1%}")

    # Get F1 scores for delay class (True/1)
    rf_f1 = rf_results['report'].get(True, rf_results['report'].get('True', {})).get('f1-score', 0)
    xgb_f1 = xgb_results['report'].get(True, xgb_results['report'].get('True', {})).get('f1-score', 0)

    print(f"\nF1-Score for Delays (better metric for imbalanced data):")
    print(f"Random Forest: {rf_f1:.1%}")
    print(f"XGBoost: {xgb_f1:.1%}")

    # Determine winner based on F1-score, NOT accuracy
    # F1 balances precision and recall, so it's better for imbalanced datasets
    if rf_f1 >= xgb_f1:
        print("\nRandom Forest performed better (or equal) on F1-score!")
        best_model = rf_model
        best_name = "random_forest"
    else:
        print("\nXGBoost performed better on F1-score!")
        best_model = xgb_model
        best_name = "xgboost"

    print()

    # Save both models
    save_model(rf_model, "random_forest_model.joblib")
    save_model(xgb_model, "xgboost_model.joblib")

    print()
    print("[DONE] Model training complete!")

    return {
        'random_forest': rf_model,
        'xgboost': xgb_model,
        'best': best_name
    }


if __name__ == "__main__":
    train_all_models()
