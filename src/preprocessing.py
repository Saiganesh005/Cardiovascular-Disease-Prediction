"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : preprocessing.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
Performs data preprocessing including:
1. Missing value handling
2. Duplicate removal
3. Feature-target separation
4. Train-test splitting
5. Feature scaling
6. Saving the scaler
==========================================================
"""

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    SCALER_FILE,
)


def remove_duplicates(df):
    """
    Remove duplicate records.
    """

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate rows...")
        df = df.drop_duplicates()

    return df


def handle_missing_values(df):
    """
    Check for missing values.
    """

    missing = df.isnull().sum()

    print("\nMissing Values")
    print("-" * 40)
    print(missing)

    if missing.sum() == 0:
        print("\nNo missing values found.")

    return df


def split_features_target(df):
    """
    Separate features and target variable.
    """

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    return X, y


def split_dataset(X, y):
    """
    Split dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nDataset Split Completed")
    print("-" * 40)
    print("Training Samples :", X_train.shape[0])
    print("Testing Samples  :", X_test.shape[0])

    return X_train, X_test, y_train, y_test


def feature_scaling(X_train, X_test):
    """
    Scale numerical features using StandardScaler.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_FILE)

    print(f"\nScaler saved to:\n{SCALER_FILE}")

    return X_train_scaled, X_test_scaled


def preprocess_data(df):
    """
    Complete preprocessing pipeline.
    """

    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Missing Values
    df = handle_missing_values(df)

    # Remove Duplicates
    df = remove_duplicates(df)

    # Split Features & Target
    X, y = split_features_target(df)

    # Train Test Split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Scaling
    X_train_scaled, X_test_scaled = feature_scaling(
        X_train,
        X_test,
    )

    print("\nPreprocessing Completed Successfully.")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )


if __name__ == "__main__":

    from src.data_loader import load_data

    dataframe = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(dataframe)

    print("\nTraining Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)
