"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : train.py
Author  : Robbi Sai Ganesh Devi Prasad

Description
-----------
Complete training pipeline for the project.

Pipeline
--------
1. Load Dataset
2. Feature Engineering
3. Data Preprocessing
4. Exploratory Data Analysis (EDA)
5. Train Machine Learning Models
6. Train Deep Learning Model
7. Evaluate Models
8. Save Models
9. Save Reports
==========================================================
"""

import warnings

warnings.filterwarnings("ignore")

from src.data_loader import load_data
from src.feature_engineering import engineer_features
from src.preprocessing import preprocess_data

from src.machine_learning import (
    train_all_models,
    best_model,
)

from src.deep_learning import (
    train_deep_learning,
)

from src.visualization import (
    generate_eda_plots,
    plot_training_accuracy,
    plot_training_loss,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_model_comparison,
)

from src.evaluation import (
    evaluate_model,
    evaluate_all_models,
    save_results,
    save_classification_report,
)

from src.logger import (
    log_info,
    log_training_start,
    log_training_end,
    log_dataset_info,
)

from sklearn.metrics import roc_curve


# ==========================================================
# Main Training Function
# ==========================================================

def main():

    print("=" * 70)
    print("CARDIOVASCULAR DISEASE PREDICTION")
    print("COMPLETE TRAINING PIPELINE")
    print("=" * 70)

    log_info("Training pipeline started.")

    # ======================================================
    # Load Dataset
    # ======================================================

    print("\nLoading Dataset...")

    dataframe = load_data()

    log_dataset_info(dataframe)

    # ======================================================
    # Feature Engineering
    # ======================================================

    print("\nFeature Engineering...")

    dataframe = engineer_features(dataframe)

    # ======================================================
    # Data Preprocessing
    # ======================================================

    print("\nPreprocessing Dataset...")

    X_train, X_test, y_train, y_test = preprocess_data(
        dataframe
    )

    # ======================================================
    # Exploratory Data Analysis
    # ======================================================

    print("\nGenerating EDA Visualizations...")

    generate_eda_plots(dataframe)

    # ======================================================
    # Machine Learning
    # ======================================================

    log_training_start("Machine Learning Models")

    models, ml_results = train_all_models(

        X_train,

        X_test,

        y_train,

        y_test,

    )

    log_training_end("Machine Learning Models")

    best_ml = best_model(ml_results)

    # ======================================================
    # Deep Learning
    # ======================================================

    log_training_start("Artificial Neural Network")

    (
        ann_model,
        history,
        ann_predictions,
        ann_probability,
        ann_metrics,
    ) = train_deep_learning(

        X_train,

        X_test,

        y_train,

        y_test,

    )

    log_training_end("Artificial Neural Network")

    # ======================================================
    # ANN Evaluation
    # ======================================================

    ann_result = evaluate_model(

        "Artificial Neural Network",

        y_test,

        ann_predictions,

        ann_probability,

    )

    # ======================================================
    # Combine Results
    # ======================================================

    all_results = ml_results.to_dict("records")

    all_results.append(ann_result)

    comparison = evaluate_all_models(all_results)

    save_results(comparison)

    compare_best = comparison.iloc[0]

    print("\nOverall Best Model")

    print("-" * 50)

    print(compare_best)

    # ======================================================
    # Save Classification Report
    # ======================================================

    save_classification_report(

        y_test,

        ann_predictions,

    )

    # ======================================================
    # Training Curves
    # ======================================================

    plot_training_accuracy(history)

    plot_training_loss(history)

    # ======================================================
    # Confusion Matrix
    # ======================================================

    plot_confusion_matrix(

        y_test,

        ann_predictions,

        "Artificial Neural Network",

    )

    # ======================================================
    # ROC Curve
    # ======================================================

    fpr, tpr, _ = roc_curve(

        y_test,

        ann_probability,

    )

    plot_roc_curve(

        fpr,

        tpr,

    )

    # ======================================================
    # Model Comparison
    # ======================================================

    plot_model_comparison(comparison)

    print("\n")

    print("=" * 70)

    print("PROJECT TRAINING COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print("\nSaved Outputs")

    print("✔ Trained Models")

    print("✔ ANN Model")

    print("✔ Standard Scaler")

    print("✔ EDA Images")

    print("✔ ROC Curve")

    print("✔ Confusion Matrix")

    print("✔ Model Comparison")

    print("✔ Classification Report")

    print("✔ CSV Performance Report")

    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()
