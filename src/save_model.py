"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : save_model.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
Utility functions for saving and loading Machine Learning
and Deep Learning models.

Supports:
1. Scikit-Learn Models (.pkl)
2. TensorFlow/Keras Models (.keras)
3. StandardScaler (.pkl)
==========================================================
"""

import joblib
import tensorflow as tf
from pathlib import Path

from src.config import (
    MODELS_DIR,
    SCALER_FILE,
)


# ==========================================================
# Create Models Directory
# ==========================================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Save Scikit-Learn Model
# ==========================================================

def save_ml_model(
    model,
    file_path,
):
    """
    Save a Machine Learning model (.pkl)

    Parameters
    ----------
    model : sklearn model
    file_path : pathlib.Path
    """

    joblib.dump(
        model,
        file_path
    )

    print(f"\nModel Saved Successfully")

    print(file_path)


# ==========================================================
# Load Scikit-Learn Model
# ==========================================================

def load_ml_model(
    file_path,
):
    """
    Load Machine Learning model.
    """

    model = joblib.load(
        file_path
    )

    print(f"\nModel Loaded Successfully")

    print(file_path)

    return model


# ==========================================================
# Save ANN Model
# ==========================================================

def save_ann_model(
    model,
    file_path,
):
    """
    Save TensorFlow/Keras model.
    """

    model.save(
        file_path
    )

    print(f"\nANN Model Saved")

    print(file_path)


# ==========================================================
# Load ANN Model
# ==========================================================

def load_ann_model(
    file_path,
):
    """
    Load TensorFlow/Keras model.
    """

    model = tf.keras.models.load_model(
        file_path
    )

    print(f"\nANN Model Loaded")

    print(file_path)

    return model


# ==========================================================
# Save Standard Scaler
# ==========================================================

def save_scaler(
    scaler,
):
    """
    Save StandardScaler.
    """

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    print("\nScaler Saved")

    print(SCALER_FILE)


# ==========================================================
# Load Standard Scaler
# ==========================================================

def load_scaler():
    """
    Load StandardScaler.
    """

    scaler = joblib.load(
        SCALER_FILE
    )

    print("\nScaler Loaded")

    print(SCALER_FILE)

    return scaler


# ==========================================================
# Save Best Model
# ==========================================================

def save_best_model(
    model,
    model_name,
):
    """
    Save best model automatically.

    Example:
    --------
    Random Forest ->
    output/models/best_random_forest.pkl
    """

    filename = (
        f"best_{model_name.lower().replace(' ', '_')}.pkl"
    )

    path = MODELS_DIR / filename

    joblib.dump(
        model,
        path
    )

    print(f"\nBest Model Saved")

    print(path)

    return path


# ==========================================================
# Check Model Exists
# ==========================================================

def model_exists(
    file_path,
):
    """
    Check whether model exists.
    """

    return Path(
        file_path
    ).exists()


# ==========================================================
# List Saved Models
# ==========================================================

def list_saved_models():
    """
    Display all saved models.
    """

    print("\nSaved Models")

    print("-" * 40)

    for model in MODELS_DIR.glob("*"):

        print(model.name)


# ==========================================================
# Delete Model
# ==========================================================

def delete_model(
    file_path,
):
    """
    Delete saved model.
    """

    file_path = Path(
        file_path
    )

    if file_path.exists():

        file_path.unlink()

        print(f"\nDeleted")

        print(file_path)

    else:

        print("\nModel Not Found")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print("MODEL MANAGEMENT MODULE")

    print("=" * 60)

    list_saved_models()
