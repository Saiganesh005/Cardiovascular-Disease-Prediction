"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : evaluation.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
Evaluate Machine Learning and Deep Learning models.

This module provides:
1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. Confusion Matrix
6. ROC Curve & AUC
7. Classification Report
8. Save evaluation reports
==========================================================
"""

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from src.config import (
    MODEL_RESULTS,
    CLASSIFICATION_REPORT,
)


# ==========================================================
# Accuracy
# ==========================================================

def calculate_accuracy(y_true, y_pred):

    return accuracy_score(y_true, y_pred)


# ==========================================================
# Precision
# ==========================================================

def calculate_precision(y_true, y_pred):

    return precision_score(y_true, y_pred)


# ==========================================================
# Recall
# ==========================================================

def calculate_recall(y_true, y_pred):

    return recall_score(y_true, y_pred)


# ==========================================================
# F1 Score
# ==========================================================

def calculate_f1(y_true, y_pred):

    return f1_score(y_true, y_pred)


# ==========================================================
# Confusion Matrix
# ==========================================================

def get_confusion_matrix(y_true, y_pred):

    return confusion_matrix(y_true, y_pred)


# ==========================================================
# Classification Report
# ==========================================================

def get_classification_report(y_true, y_pred):

    return classification_report(y_true, y_pred)


# ==========================================================
# ROC Curve
# ==========================================================

def get_roc_curve(y_true, y_probability):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_probability
    )

    auc_score = roc_auc_score(
        y_true,
        y_probability
    )

    return fpr, tpr, auc_score


# ==========================================================
# Complete Evaluation
# ==========================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    y_probability=None,
):
    """
    Evaluate one model.
    """

    results = {

        "Model": model_name,

        "Accuracy":
            calculate_accuracy(
                y_true,
                y_pred
            ),

        "Precision":
            calculate_precision(
                y_true,
                y_pred
            ),

        "Recall":
            calculate_recall(
                y_true,
                y_pred
            ),

        "F1 Score":
            calculate_f1(
                y_true,
                y_pred
            )

    }

    if y_probability is not None:

        auc = roc_auc_score(
            y_true,
            y_probability
        )

        results["ROC AUC"] = auc

    return results


# ==========================================================
# Evaluate Multiple Models
# ==========================================================

def evaluate_all_models(results):

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(

        by="Accuracy",

        ascending=False

    )

    return results_df


# ==========================================================
# Save Results
# ==========================================================

def save_results(results_df):

    results_df.to_csv(

        MODEL_RESULTS,

        index=False

    )

    print(f"\nResults saved to:\n{MODEL_RESULTS}")


# ==========================================================
# Save Classification Report
# ==========================================================

def save_classification_report(
    y_true,
    y_pred,
):

    report = classification_report(
        y_true,
        y_pred
    )

    with open(
        CLASSIFICATION_REPORT,
        "w"
    ) as file:

        file.write(report)

    print(
        f"\nClassification Report saved to:\n{CLASSIFICATION_REPORT}"
    )


# ==========================================================
# Print Evaluation
# ==========================================================

def print_results(results):

    print("=" * 60)

    print("MODEL PERFORMANCE")

    print("=" * 60)

    for key, value in results.items():

        if isinstance(value, float):

            print(f"{key:<15}: {value:.4f}")

        else:

            print(f"{key:<15}: {value}")

    print("=" * 60)


# ==========================================================
# Compare Models
# ==========================================================

def compare_models(results_df):

    print("\nModel Comparison")

    print("-" * 60)

    print(results_df)

    print("-" * 60)

    best = results_df.iloc[0]

    print("\nBest Model")

    print(f"Model    : {best['Model']}")

    print(f"Accuracy : {best['Accuracy']:.4f}")

    return best


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("Evaluation Module Loaded Successfully.")
