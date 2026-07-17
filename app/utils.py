"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : utils.py
Author  : Robbi Sai Ganesh Devi Prasad

Description
-----------
Utility functions for the Flask web application.

Features
--------
1. Validate user input
2. Convert HTML form to model input
3. Calculate BMI
4. Convert prediction to readable text
5. Format confidence score
==========================================================
"""

import numpy as np


# ==========================================================
# Validate Input
# ==========================================================

def validate_input(form_data):
    """
    Validate user input from HTML form.

    Parameters
    ----------
    form_data : dict

    Returns
    -------
    (bool, message)
    """

    try:

        age = float(form_data["age"])
        height = float(form_data["height"])
        weight = float(form_data["weight"])
        ap_hi = float(form_data["ap_hi"])
        ap_lo = float(form_data["ap_lo"])

        if age <= 0:
            return False, "Age must be greater than zero."

        if height <= 0:
            return False, "Height must be greater than zero."

        if weight <= 0:
            return False, "Weight must be greater than zero."

        if ap_hi <= 0 or ap_lo <= 0:
            return False, "Blood pressure values must be greater than zero."

        if ap_hi < ap_lo:
            return False, "Systolic pressure must be greater than Diastolic pressure."

        return True, "Valid"

    except Exception as e:

        return False, str(e)


# ==========================================================
# Convert HTML Form to Feature Vector
# ==========================================================

def form_to_features(form_data):
    """
    Convert Flask form into model input.
    """

    features = [

        float(form_data["age"]),
        int(form_data["gender"]),
        float(form_data["height"]),
        float(form_data["weight"]),
        float(form_data["ap_hi"]),
        float(form_data["ap_lo"]),
        int(form_data["cholesterol"]),
        int(form_data["glucose"]),
        int(form_data["smoke"]),
        int(form_data["alcohol"]),
        int(form_data["active"])

    ]

    return features


# ==========================================================
# Calculate BMI
# ==========================================================

def calculate_bmi(height, weight):
    """
    Calculate Body Mass Index.

    Height : cm
    Weight : kg
    """

    height = height / 100

    bmi = weight / (height ** 2)

    return round(bmi, 2)


# ==========================================================
# Prediction Text
# ==========================================================

def prediction_text(prediction):
    """
    Convert numeric prediction into text.
    """

    if prediction == 1:

        return "Heart Disease Detected"

    return "No Heart Disease"


# ==========================================================
# Risk Level
# ==========================================================

def risk_level(confidence):
    """
    Determine risk level from confidence.
    """

    if confidence >= 0.90:
        return "Very High"

    elif confidence >= 0.75:
        return "High"

    elif confidence >= 0.50:
        return "Moderate"

    return "Low"


# ==========================================================
# Confidence Percentage
# ==========================================================

def confidence_percentage(confidence):
    """
    Convert probability to percentage.
    """

    return f"{confidence * 100:.2f}%"


# ==========================================================
# Patient Summary
# ==========================================================

def patient_summary(form_data):
    """
    Create summary of patient information.
    """

    bmi = calculate_bmi(

        float(form_data["height"]),

        float(form_data["weight"])

    )

    summary = {

        "Age": form_data["age"],

        "Gender":
            "Male"
            if form_data["gender"] == "2"
            else "Female",

        "Height": f"{form_data['height']} cm",

        "Weight": f"{form_data['weight']} kg",

        "BMI": bmi,

        "Blood Pressure":
            f"{form_data['ap_hi']} / {form_data['ap_lo']} mmHg",

        "Cholesterol": form_data["cholesterol"],

        "Glucose": form_data["glucose"],

        "Smoking":
            "Yes"
            if form_data["smoke"] == "1"
            else "No",

        "Alcohol":
            "Yes"
            if form_data["alcohol"] == "1"
            else "No",

        "Physical Activity":
            "Active"
            if form_data["active"] == "1"
            else "Inactive",

    }

    return summary


# ==========================================================
# Convert List to NumPy Array
# ==========================================================

def to_numpy(features):
    """
    Convert list into NumPy array.
    """

    return np.array(features).reshape(1, -1)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    sample = {

        "age": "18393",
        "gender": "2",
        "height": "168",
        "weight": "62",
        "ap_hi": "120",
        "ap_lo": "80",
        "cholesterol": "1",
        "glucose": "1",
        "smoke": "0",
        "alcohol": "0",
        "active": "1"

    }

    valid, message = validate_input(sample)

    print("Validation :", valid)

    print("Message    :", message)

    print()

    print("BMI :", calculate_bmi(168, 62))

    print()

    print(patient_summary(sample))
