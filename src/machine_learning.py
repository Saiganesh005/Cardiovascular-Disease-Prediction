from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier()

    model.fit(X_train, y_train)
"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : machine_learning.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
Train, evaluate and save Machine Learning models.

Algorithms:
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. Support Vector Machine
==========================================================
"""

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import (
    RANDOM_STATE,
    LOGISTIC_MODEL,
    KNN_MODEL,
    DECISION_TREE_MODEL,
    RANDOM_FOREST_MODEL,
    SVM_MODEL,
    RF_ESTIMATORS,
    RF_MAX_DEPTH,
    KNN_NEIGHBORS,
    SVM_KERNEL,
)

# ==========================================================
# Logistic Regression
# ==========================================================

def train_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test,
):

    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    joblib.dump(model, LOGISTIC_MODEL)

    return model, predictions


# ==========================================================
# KNN
# ==========================================================

def train_knn(
    X_train,
    X_test,
    y_train,
    y_test,
):

    model = KNeighborsClassifier(
        n_neighbors=KNN_NEIGHBORS
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    joblib.dump(model, KNN_MODEL)

    return model, predictions


# ==========================================================
# Decision Tree
# ==========================================================

def train_decision_tree(
    X_train,
    X_test,
    y_train,
    y_test,
):

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    joblib.dump(model, DECISION_TREE_MODEL)

    return model, predictions


# ==========================================================
# Random Forest
# ==========================================================

def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
):

    model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    joblib.dump(model, RANDOM_FOREST_MODEL)

    return model, predictions


# ==========================================================
# Support Vector Machine
# ==========================================================

def train_svm(
    X_train,
    X_test,
    y_train,
    y_test,
):

    model = SVC(
        kernel=SVM_KERNEL,
        probability=True,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    joblib.dump(model, SVM_MODEL)

    return model, predictions


# ==========================================================
# Evaluation Metrics
# ==========================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    return {

        "Accuracy":
            accuracy_score(y_true, y_pred),

        "Precision":
            precision_score(y_true, y_pred),

        "Recall":
            recall_score(y_true, y_pred),

        "F1 Score":
            f1_score(y_true, y_pred),

    }


# ==========================================================
# Train All Models
# ==========================================================

def train_all_models(
    X_train,
    X_test,
    y_train,
    y_test,
):

    print("=" * 60)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 60)

    results = []

    models = {}

    # ----------------------------------------------

    print("\nTraining Logistic Regression...")

    model, pred = train_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    models["Logistic Regression"] = model

    metrics = calculate_metrics(y_test, pred)

    metrics["Model"] = "Logistic Regression"

    results.append(metrics)

    # ----------------------------------------------

    print("Training KNN...")

    model, pred = train_knn(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    models["KNN"] = model

    metrics = calculate_metrics(y_test, pred)

    metrics["Model"] = "KNN"

    results.append(metrics)

    # ----------------------------------------------

    print("Training Decision Tree...")

    model, pred = train_decision_tree(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    models["Decision Tree"] = model

    metrics = calculate_metrics(y_test, pred)

    metrics["Model"] = "Decision Tree"

    results.append(metrics)

    # ----------------------------------------------

    print("Training Random Forest...")

    model, pred = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    models["Random Forest"] = model

    metrics = calculate_metrics(y_test, pred)

    metrics["Model"] = "Random Forest"

    results.append(metrics)

    # ----------------------------------------------

    print("Training Support Vector Machine...")

    model, pred = train_svm(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    models["Support Vector Machine"] = model

    metrics = calculate_metrics(y_test, pred)

    metrics["Model"] = "Support Vector Machine"

    results.append(metrics)

    # ----------------------------------------------

    results = pd.DataFrame(results)

    results = results[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ]
    ]

    results = results.sort_values(
        by="Accuracy",
        ascending=False,
    )

    print("\n")
    print(results)

    print("\nTraining Completed Successfully.")

    return models, results


# ==========================================================
# Best Model
# ==========================================================

def best_model(results):

    best = results.iloc[0]

    print("\nBest Model")
    print("-" * 40)

    print(f"Model    : {best['Model']}")
    print(f"Accuracy : {best['Accuracy']:.4f}")

    return best


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    from src.data_loader import load_data
    from src.preprocessing import preprocess_data

    dataframe = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(
        dataframe
    )

    models, results = train_all_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    best_model(results)
    return model
