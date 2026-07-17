"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : prediction.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
Prediction module for Machine Learning and Deep Learning
models.

Features
--------
1. Load saved ML models
2. Load ANN model
3. Load StandardScaler
4. Predict Heart Disease
5. Predict Probability
==========================================================
"""

import numpy as np

from src.config import (
    RANDOM_FOREST_MODEL,
    ANN_MODEL,
)

from src.save_model import (
    load_ml_model,
    load_ann_model,
    load_scaler,
)


# ==========================================================
# Preprocess Input
# ==========================================================

def preprocess_input(data):
    """
    Scale user input.

    Parameters
    ----------
    data : list

    Returns
    -------
    numpy.ndarray
    """

    scaler = load_scaler()

    data = np.array(data).reshape(1, -1)

    data = scaler.transform(data)

    return data


# ==========================================================
# Predict using ML Model
# ==========================================================

def predict_ml(data):
    """
    Predict using Random Forest model.

    Returns
    -------
    prediction
    probability
    """

    model = load_ml_model(
        RANDOM_FOREST_MODEL
    )

    data = preprocess_input(data)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0]

    confidence = max(probability)

    return prediction, confidence


# ==========================================================
# Predict using ANN
# ==========================================================

def predict_ann(data):
    """
    Predict using ANN model.
    """

    model = load_ann_model(
        ANN_MODEL
    )

    data = preprocess_input(data)

    probability = model.predict(
        data,
        verbose=0
    )[0][0]

    prediction = int(
        probability >= 0.50
    )

    confidence = probability if prediction == 1 else 1 - probability

    return prediction, confidence


# ==========================================================
# Display Prediction
# ==========================================================

def print_prediction(
    prediction,
    confidence,
):
    """
    Print prediction nicely.
    """

    print("=" * 50)

    print("CARDIOVASCULAR DISEASE PREDICTION")

    print("=" * 50)

    if prediction == 1:

        print("Prediction : Heart Disease Detected")

    else:

        print("Prediction : No Heart Disease")

    print(f"Confidence : {confidence:.2%}")

    print("=" * 50)


# ==========================================================
# Predict from Feature Dictionary
# ==========================================================

def predict_from_dict(patient):
    """
    Predict from dictionary input.

    Example
    -------
    {
        "age":18393,
        "gender":2,
        ...
    }
    """

    values = list(patient.values())

    prediction, confidence = predict_ml(values)

    return prediction, confidence


# ==========================================================
# Example Patient
# ==========================================================

example_patient = [

    18393,   # age

    2,       # gender

    168,     # height

    62,      # weight

    110,     # ap_hi

    80,      # ap_lo

    0,       # cholesterol

    0,       # glucose

    0,       # smoke

    0,       # alcohol

    1        # active

]


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nRandom Forest Prediction\n")

    prediction, confidence = predict_ml(
        example_patient
    )

    print_prediction(
        prediction,
        confidence,
    )

    print("\nANN Prediction\n")

    prediction, confidence = predict_ann(
        example_patient
    )

    print_prediction(
        prediction,
        confidence,
    )
