"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : predict.py
Author  : Robbi Sai Ganesh Devi Prasad

Description
-----------
Prediction service for the Flask application.

Features
--------
1. Predict using Random Forest
2. Predict using ANN
3. Predict using both models
4. Generate prediction summary
==========================================================
"""

from src.prediction import (
    predict_ml,
    predict_ann,
)

from app.utils import (
    prediction_text,
    confidence_percentage,
    risk_level,
)


# ==========================================================
# Random Forest Prediction
# ==========================================================

def random_forest_prediction(patient):

    prediction, confidence = predict_ml(patient)

    return {

        "model": "Random Forest",

        "prediction": prediction,

        "result": prediction_text(prediction),

        "confidence": confidence,

        "confidence_percentage":
            confidence_percentage(confidence),

        "risk_level":
            risk_level(confidence)

    }


# ==========================================================
# ANN Prediction
# ==========================================================

def ann_prediction(patient):

    prediction, confidence = predict_ann(patient)

    return {

        "model": "Artificial Neural Network",

        "prediction": prediction,

        "result": prediction_text(prediction),

        "confidence": confidence,

        "confidence_percentage":
            confidence_percentage(confidence),

        "risk_level":
            risk_level(confidence)

    }


# ==========================================================
# Predict using Both Models
# ==========================================================

def predict(patient):
    """
    Predict using both ML and DL models.
    """

    rf = random_forest_prediction(patient)

    ann = ann_prediction(patient)

    return {

        "Random Forest": rf,

        "Artificial Neural Network": ann

    }


# ==========================================================
# Final Decision
# ==========================================================

def final_prediction(results):
    """
    Majority voting between ML and ANN.
    """

    rf = results["Random Forest"]["prediction"]

    ann = results["Artificial Neural Network"]["prediction"]

    positive = rf + ann

    if positive >= 1:

        final = "Heart Disease Detected"

    else:

        final = "No Heart Disease"

    return final


# ==========================================================
# Generate Complete Report
# ==========================================================

def prediction_report(patient):

    predictions = predict(patient)

    report = {

        "Final Prediction":
            final_prediction(predictions),

        "Random Forest":
            predictions["Random Forest"],

        "Artificial Neural Network":
            predictions["Artificial Neural Network"]

    }

    return report


# ==========================================================
# Print Report
# ==========================================================

def print_report(report):

    print("=" * 60)

    print("CARDIOVASCULAR DISEASE PREDICTION REPORT")

    print("=" * 60)

    print("\nFinal Result")

    print("------------------------")

    print(report["Final Prediction"])

    print("\n")

    for model in [

        "Random Forest",

        "Artificial Neural Network"

    ]:

        result = report[model]

        print(model)

        print("-" * 30)

        print("Prediction :", result["result"])

        print("Confidence :", result["confidence_percentage"])

        print("Risk Level :", result["risk_level"])

        print()


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    patient = [

        18393,
        2,
        168,
        62,
        120,
        80,
        1,
        1,
        0,
        0,
        1

    ]

    report = prediction_report(patient)

    print_report(report)
