"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : feature_engineering.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
This module performs feature engineering operations:
1. Remove unnecessary columns
2. Create new features
3. Feature selection
4. Encode categorical variables
5. Feature importance
==========================================================
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import TARGET_COLUMN, RANDOM_STATE


# ==========================================================
# Remove Unnecessary Columns
# ==========================================================

def drop_columns(df, columns=None):
    """
    Remove unwanted columns from the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
    columns : list

    Returns
    -------
    pandas.DataFrame
    """

    if columns is None:
        return df

    existing_columns = [col for col in columns if col in df.columns]

    if existing_columns:
        df = df.drop(columns=existing_columns)

        print(f"Removed Columns : {existing_columns}")

    return df


# ==========================================================
# Create BMI Feature
# ==========================================================

def create_bmi(df):
    """
    Create BMI feature.

    BMI = Weight / Height²
    """

    if "height" in df.columns and "weight" in df.columns:

        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

        print("BMI feature created.")

    return df


# ==========================================================
# Blood Pressure Difference
# ==========================================================

def create_bp_difference(df):
    """
    Difference between Systolic and Diastolic Pressure.
    """

    if "ap_hi" in df.columns and "ap_lo" in df.columns:

        df["bp_difference"] = df["ap_hi"] - df["ap_lo"]

        print("Blood Pressure Difference feature created.")

    return df


# ==========================================================
# Pulse Pressure
# ==========================================================

def create_pulse_pressure(df):
    """
    Pulse Pressure = Systolic - Diastolic
    """

    if "ap_hi" in df.columns and "ap_lo" in df.columns:

        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

        print("Pulse Pressure feature created.")

    return df


# ==========================================================
# Age in Years
# ==========================================================

def convert_age(df):
    """
    Convert age from days to years.
    """

    if "age" in df.columns:

        df["age_years"] = (df["age"] / 365).round(1)

        print("Age converted to years.")

    return df


# ==========================================================
# Feature Importance
# ==========================================================

def feature_importance(df):
    """
    Calculate feature importance using Random Forest.
    """

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(X, y)

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    return importance


# ==========================================================
# Select Top Features
# ==========================================================

def select_top_features(df, top_n=10):
    """
    Select top N important features.
    """

    importance = feature_importance(df)

    selected = importance.head(top_n)["Feature"].tolist()

    selected.append(TARGET_COLUMN)

    print(f"Top {top_n} Features Selected")

    return df[selected]


# ==========================================================
# Complete Feature Engineering Pipeline
# ==========================================================

def engineer_features(df):
    """
    Perform all feature engineering operations.
    """

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = convert_age(df)

    df = create_bmi(df)

    df = create_bp_difference(df)

    df = create_pulse_pressure(df)

    print("\nFeature Engineering Completed Successfully.")

    return df


# ==========================================================
# Main Function
# ==========================================================

if __name__ == "__main__":

    from src.data_loader import load_data

    dataframe = load_data()

    dataframe = engineer_features(dataframe)

    print("\nNew Features Added")

    print(dataframe.head())

    importance = feature_importance(dataframe)

    print("\nFeature Importance")

    print(importance.head(10))
