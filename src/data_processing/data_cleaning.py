"""
Data Cleaning
Handles missing data, outliers, and data quality issues

WHY DATA CLEANING MATTERS:
Real-world data is messy! We need to:
- Handle missing values (what if weather API fails?)
- Remove outliers (a 500 minute delay is probably an error)
- Standardize formats (different date formats, etc.)
- Validate data quality

COMMON ISSUES WE'LL HANDLE:
1. Missing data (fill with averages or remove)
2. Duplicate records
3. Invalid values (negative delays, impossible timestamps)
4. Inconsistent formats
"""

import pandas as pd
import numpy as np


def handle_missing_values(df):
    """
    Handle missing values in the dataset

    Args:
        df: DataFrame with potentially missing values

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # TODO: Implement missing value handling
    # Options:
    # - Fill with mean/median for numerical data
    # - Fill with mode for categorical data
    # - Drop rows with too many missing values
    # - Forward fill for time-series data

    print("🧹 Handling missing values...")
    pass


def remove_outliers(df, column, threshold=3):
    """
    Remove statistical outliers (beyond 3 standard deviations)

    Args:
        df: DataFrame
        column: Column name to check for outliers
        threshold: Standard deviations threshold

    Returns:
        pd.DataFrame: DataFrame without outliers
    """
    # TODO: Detect and remove outliers
    # Use z-score or IQR method
    pass


def validate_data_types(df):
    """
    Ensure all columns have correct data types

    Args:
        df: DataFrame to validate

    Returns:
        pd.DataFrame: DataFrame with corrected types
    """
    # TODO: Convert columns to proper types
    # - Timestamps to datetime
    # - Delays to float
    # - Categorical variables to category type
    pass


def remove_duplicates(df):
    """
    Remove duplicate records

    Args:
        df: DataFrame potentially with duplicates

    Returns:
        pd.DataFrame: Deduplicated DataFrame
    """
    # TODO: Remove duplicate rows
    print("🧹 Removing duplicates...")
    pass


def clean_dataset(df):
    """
    Master cleaning function - applies all cleaning steps

    Args:
        df: Raw DataFrame

    Returns:
        pd.DataFrame: Fully cleaned DataFrame
    """
    # TODO: Apply all cleaning steps in order
    # 1. Remove duplicates
    # 2. Validate data types
    # 3. Handle missing values
    # 4. Remove outliers
    # 5. Final validation

    print("🧹 Starting data cleaning pipeline...")
    return df


if __name__ == "__main__":
    print("Data cleaning module")
    print("This ensures data quality before ML training")
