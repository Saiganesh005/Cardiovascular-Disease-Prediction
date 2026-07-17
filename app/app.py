"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : app.py
Author  : Robbi Sai Ganesh Devi Prasad

Description
-----------
Flask Web Application for predicting Cardiovascular Disease.
==========================================================
"""

from flask import Flask, render_template, request
import numpy as np

from src.prediction import predict_ml, predict_ann

app = Flask(__name__)


# ==========================================================
# Home Page
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================================
# Prediction
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        age = float(request.form["age"])
        gender = int(request.form["gender"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = float(request.form["ap_hi"])
        ap_lo = float(request.form["ap_lo"])
        cholesterol = int(request.form["cholesterol"])
        glucose = int(request.form["glucose"])
        smoke = int(request.form["smoke"])
        alcohol = int(request.form["alcohol"])
        active = int(request.form["active"])

        patient = [

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

        # -----------------------------
        # Random Forest Prediction
        # -----------------------------

        rf_prediction, rf_confidence = predict_ml(patient)

        # -----------------------------
        # ANN Prediction
        # -----------------------------

        ann_prediction, ann_confidence = predict_ann(patient)

        rf_result = (
            "Heart Disease Detected"
            if rf_prediction == 1
            else "No Heart Disease"
        )

        ann_result = (
            "Heart Disease Detected"
            if ann_prediction == 1
            else "No Heart Disease"
        )

        return render_template(

            "result.html",

            rf_result=rf_result,
            rf_confidence=round(rf_confidence * 100, 2),

            ann_result=ann_result,
            ann_confidence=round(ann_confidence * 100, 2)

        )

    except Exception as e:

        return render_template(

            "error.html",

            error=str(e)

        )


# ==========================================================
# Run Application
# ==========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
