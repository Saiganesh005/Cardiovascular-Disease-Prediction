"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : config.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
This file stores all project-wide constants, directory
paths, filenames, and model configuration values.
==========================================================
"""

from pathlib import Path

# ==========================================================
# Project Root Directory
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Dataset Paths
# ==========================================================

DATASET_DIR = PROJECT_ROOT / "dataset"

DATASET_FILE = DATASET_DIR / "cardio_train.csv"

# ==========================================================
# Output Directories
# ==========================================================

OUTPUT_DIR = PROJECT_ROOT / "output"

IMAGES_DIR = OUTPUT_DIR / "images"

MODELS_DIR = OUTPUT_DIR / "models"

REPORTS_DIR = OUTPUT_DIR / "reports"

LOGS_DIR = OUTPUT_DIR / "logs"

# ==========================================================
# Documentation
# ==========================================================

DOCS_DIR = PROJECT_ROOT / "docs"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ==========================================================
# Model Files
# ==========================================================

LOGISTIC_MODEL = MODELS_DIR / "logistic_regression.pkl"

KNN_MODEL = MODELS_DIR / "knn.pkl"

DECISION_TREE_MODEL = MODELS_DIR / "decision_tree.pkl"

RANDOM_FOREST_MODEL = MODELS_DIR / "random_forest.pkl"

SVM_MODEL = MODELS_DIR / "svm.pkl"

ANN_MODEL = MODELS_DIR / "heart_disease_ann.keras"

SCALER_FILE = MODELS_DIR / "scaler.pkl"

# ==========================================================
# Reports
# ==========================================================

MODEL_RESULTS = REPORTS_DIR / "model_results.csv"

CLASSIFICATION_REPORT = REPORTS_DIR / "classification_report.txt"

PREDICTIONS = REPORTS_DIR / "prediction_results.csv"

# ==========================================================
# Dataset Configuration
# ==========================================================

TARGET_COLUMN = "cardio"

TEST_SIZE = 0.20

RANDOM_STATE = 42

VALIDATION_SPLIT = 0.20

# ==========================================================
# Deep Learning Parameters
# ==========================================================

EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 0.001

PATIENCE = 10

# ==========================================================
# Image Configuration
# ==========================================================

FIGURE_SIZE = (10, 6)

DPI = 300

IMAGE_FORMAT = "png"

# ==========================================================
# Random Forest Parameters
# ==========================================================

RF_ESTIMATORS = 100

RF_MAX_DEPTH = None

# ==========================================================
# SVM Parameters
# ==========================================================

SVM_KERNEL = "rbf"

# ==========================================================
# KNN Parameters
# ==========================================================

KNN_NEIGHBORS = 5

# ==========================================================
# Create Required Directories
# ==========================================================

DIRECTORIES = [
    DATASET_DIR,
    OUTPUT_DIR,
    IMAGES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    DOCS_DIR,
    NOTEBOOKS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================================
# End of File
# ==========================================================
