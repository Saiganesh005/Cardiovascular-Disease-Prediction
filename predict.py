"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : predict.py
Author  : Robbi Sai Ganesh Devi Prasad

Description
-----------
Prediction script for the trained Cardiovascular Disease
Prediction models.

This script:
1. Loads the trained Random Forest model
2. Loads the trained ANN model
3. Loads the StandardScaler
4. Accepts patient information
5. Predicts cardiovascular disease
6. Displays prediction confidence
==========================================================
"""

import numpy as np

from src.prediction import (
    predict_ml,
    predict_ann,
    print_prediction,
)

# ==========================================================
# Input Patient Details
# ==========================================================

def get_patient_data():
    """
    Accept patient details from the user.
    """

    print("\nEnter Patient Details\n")

    age = float(input("Age (days): "))
    gender = int(input("Gender (1=Female, 2=Male): "))
    height = float(input("Height (cm): "))
    weight = float(input("Weight (kg): "))
    ap_hi = float(input("Systolic BP: "))
    ap_lo = float(input("Diastolic BP: "))
    cholesterol = int(input("Cholesterol (1-3): "))
    glucose = int(input("Glucose (1-3): "))
    smoke = int(input("Smoke (0/1): "))
    alcohol = int(input("Alcohol (0/1): "))
    active = int(input("Physical Activity (0/1): "))

    return [

        age,

        gender,

        height,

        weight,

        ap_hi,

        ap_lo,

        cholesterol,

        glucose,

        smoke,

        alcohol,

        active

    ]


# ==========================================================
# Sample Patient
# ==========================================================

def sample_patient():
    """
    Sample patient for quick testing.
    """

    return [

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


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 65)
    print("CARDIOVASCULAR DISEASE PREDICTION SYSTEM")
    print("=" * 65)

    print("\nChoose Input Method")

    print("1. Manual Input")

    print("2. Sample Patient")

    choice = input("\nEnter choice (1/2): ")

    if choice == "1":

        patient = get_patient_data()

    else:

        patient = sample_patient()

    print("\n")

    print("=" * 65)

    print("Random Forest Prediction")

    print("=" * 65)

    prediction, confidence = predict_ml(patient)

    print_prediction(

        prediction,

        confidence,

    )

    print("\n")

    print("=" * 65)

    print("Artificial Neural Network Prediction")

    print("=" * 65)

    prediction, confidence = predict_ann(patient)

    print_prediction(

        prediction,

        confidence,

    )

    print("\nPrediction Completed Successfully.")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()
