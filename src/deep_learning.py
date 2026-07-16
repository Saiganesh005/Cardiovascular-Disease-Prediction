"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : deep_learning.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
Builds, trains, evaluates and saves an Artificial
Neural Network (ANN) using TensorFlow/Keras.
==========================================================
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import (
    ANN_MODEL,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    PATIENCE,
)


# ==========================================================
# Build ANN
# ==========================================================

def build_model(input_dim):
    """
    Build Artificial Neural Network.
    """

    model = Sequential([

        Dense(
            64,
            activation="relu",
            input_shape=(input_dim,)
        ),

        Dropout(0.30),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.20),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1,
            activation="sigmoid"
        )

    ])

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# Train Model
# ==========================================================

def train_ann(
    X_train,
    y_train,
):
    """
    Train ANN model.
    """

    model = build_model(
        X_train.shape[1]
    )

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=PATIENCE,

        restore_best_weights=True,

        verbose=1

    )

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.20,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[early_stop],

        verbose=1

    )

    return model, history


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_ann(
    model,
    X_test,
    y_test,
):
    """
    Evaluate trained ANN.
    """

    probability = model.predict(
        X_test,
        verbose=0
    )

    predictions = (
        probability > 0.5
    ).astype(int)

    metrics = {

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions
            ),

        "Recall":
            recall_score(
                y_test,
                predictions
            ),

        "F1 Score":
            f1_score(
                y_test,
                predictions
            )

    }

    return metrics, predictions, probability


# ==========================================================
# Save Model
# ==========================================================

def save_ann(model):
    """
    Save trained ANN model.
    """

    model.save(ANN_MODEL)

    print(f"\nModel saved successfully.")

    print(ANN_MODEL)


# ==========================================================
# Load Model
# ==========================================================

def load_ann():
    """
    Load saved ANN model.
    """

    model = tf.keras.models.load_model(
        ANN_MODEL
    )

    return model


# ==========================================================
# Predict New Sample
# ==========================================================

def predict(
    model,
    sample,
):
    """
    Predict new patient.
    """

    probability = model.predict(
        sample,
        verbose=0
    )[0][0]

    prediction = int(
        probability >= 0.50
    )

    return prediction, probability


# ==========================================================
# Complete Pipeline
# ==========================================================

def train_deep_learning(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Complete ANN pipeline.
    """

    print("=" * 60)
    print("TRAINING DEEP LEARNING MODEL")
    print("=" * 60)

    model, history = train_ann(
        X_train,
        y_train,
    )

    metrics, predictions, probability = evaluate_ann(
        model,
        X_test,
        y_test,
    )

    save_ann(model)

    print("\nANN Performance")

    print("-" * 40)

    for key, value in metrics.items():

        print(f"{key:<12}: {value:.4f}")

    return (

        model,

        history,

        predictions,

        probability,

        metrics,

    )


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

    model, history, pred, prob, metrics = train_deep_learning(

        X_train,

        X_test,

        y_train,

        y_test,

    )

    sample = X_test[:1]

    prediction, probability = predict(
        model,
        sample,
    )

    print("\nSample Prediction")

    print("-------------------------")

    print("Prediction :", prediction)

    print("Probability:", round(probability, 4))
