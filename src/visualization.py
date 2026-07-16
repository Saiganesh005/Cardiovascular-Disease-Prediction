"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : visualization.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
This module generates and saves all project
visualizations into output/images/.
==========================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.config import (
    IMAGES_DIR,
    FIGURE_SIZE,
    DPI,
)


# ---------------------------------------------------------
# Plot Style
# ---------------------------------------------------------

sns.set_style("whitegrid")


# ---------------------------------------------------------
# Helper Function
# ---------------------------------------------------------

def save_plot(filename):
    """
    Save the current matplotlib figure.
    """

    plt.tight_layout()

    plt.savefig(
        IMAGES_DIR / filename,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------------
# Histogram
# ---------------------------------------------------------

def plot_histograms(df):

    df.hist(figsize=(18, 12))

    save_plot("histogram.png")

    print("✔ histogram.png saved")


# ---------------------------------------------------------
# Target Distribution
# ---------------------------------------------------------

def plot_target_distribution(df):

    plt.figure(figsize=FIGURE_SIZE)

    sns.countplot(
        data=df,
        x="cardio"
    )

    plt.title("Heart Disease Distribution")

    save_plot("countplot.png")

    print("✔ countplot.png saved")


# ---------------------------------------------------------
# Box Plot
# ---------------------------------------------------------

def plot_boxplot(df):

    plt.figure(figsize=(15, 8))

    sns.boxplot(data=df)

    plt.xticks(rotation=45)

    plt.title("Feature Box Plot")

    save_plot("boxplot.png")

    print("✔ boxplot.png saved")


# ---------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------

def plot_correlation(df):

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    save_plot("correlation_heatmap.png")

    print("✔ correlation_heatmap.png saved")


# ---------------------------------------------------------
# Pair Plot
# ---------------------------------------------------------

def plot_pairplot(df):

    columns = [
        "age",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cardio"
    ]

    pair = sns.pairplot(
        df[columns],
        hue="cardio"
    )

    pair.savefig(
        IMAGES_DIR / "pairplot.png",
        dpi=DPI
    )

    plt.close()

    print("✔ pairplot.png saved")


# ---------------------------------------------------------
# Training Accuracy
# ---------------------------------------------------------

def plot_training_accuracy(history):

    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(
        history.history["accuracy"],
        label="Training"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation"
    )

    plt.title("Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    save_plot("training_accuracy.png")

    print("✔ training_accuracy.png saved")


# ---------------------------------------------------------
# Training Loss
# ---------------------------------------------------------

def plot_training_loss(history):

    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(
        history.history["loss"],
        label="Training"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation"
    )

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    save_plot("training_loss.png")

    print("✔ training_loss.png saved")


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    save_plot("confusion_matrix.png")

    print("✔ confusion_matrix.png saved")


# ---------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------

def plot_roc_curve(y_true, y_probability):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_probability
    )

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    save_plot("roc_curve.png")

    print("✔ roc_curve.png saved")


# ---------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------

def plot_model_comparison(scores):

    plt.figure(figsize=(10, 6))

    plt.bar(
        scores["Model"],
        scores["Accuracy"]
    )

    plt.xticks(rotation=20)

    plt.ylabel("Accuracy")

    plt.title("Model Comparison")

    save_plot("model_comparison.png")

    print("✔ model_comparison.png saved")


# ---------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------

def plot_feature_importance(model, feature_names):

    if not hasattr(model, "feature_importances_"):
        return

    importance = model.feature_importances_

    plt.figure(figsize=(10, 8))

    sns.barplot(
        x=importance,
        y=feature_names
    )

    plt.title("Feature Importance")

    save_plot("feature_importance.png")

    print("✔ feature_importance.png saved")


# ---------------------------------------------------------
# Generate All EDA Plots
# ---------------------------------------------------------

def generate_eda_plots(df):

    print("\nGenerating EDA Visualizations...\n")

    plot_histograms(df)

    plot_target_distribution(df)

    plot_boxplot(df)

    plot_correlation(df)

    plot_pairplot(df)

    print("\nEDA Visualization Completed.")
