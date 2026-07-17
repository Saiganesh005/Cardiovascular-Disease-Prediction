"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : logger.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
Logging utility for the Cardiovascular Disease Prediction
project.

Features
--------
1. Console Logging
2. File Logging
3. Error Logging
4. Training Logs
5. Prediction Logs
==========================================================
"""

import logging
from pathlib import Path
from datetime import datetime

from src.config import LOGS_DIR

# ==========================================================
# Create Logs Directory
# ==========================================================

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# Log File Name
# ==========================================================

LOG_FILE = LOGS_DIR / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ==========================================================
# Logger Configuration
# ==========================================================

logger = logging.getLogger("CardioPredict")

logger.setLevel(logging.INFO)

# Avoid duplicate handlers
if not logger.handlers:

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)-8s | %(message)s",

        "%Y-%m-%d %H:%M:%S"

    )

    # ---------------- Console Handler ---------------- #

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(formatter)

    # ---------------- File Handler ---------------- #

    file_handler = logging.FileHandler(LOG_FILE)

    file_handler.setLevel(logging.INFO)

    file_handler.setFormatter(formatter)

    # ---------------- Add Handlers ---------------- #

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)


# ==========================================================
# Logging Functions
# ==========================================================

def log_info(message):
    """
    Log information.
    """
    logger.info(message)


def log_warning(message):
    """
    Log warning.
    """
    logger.warning(message)


def log_error(message):
    """
    Log error.
    """
    logger.error(message)


def log_debug(message):
    """
    Log debug information.
    """
    logger.debug(message)


def log_exception(exception):
    """
    Log exception with traceback.
    """
    logger.exception(exception)


# ==========================================================
# Training Logs
# ==========================================================

def log_training_start(model_name):
    logger.info("=" * 60)
    logger.info(f"Training Started : {model_name}")
    logger.info("=" * 60)


def log_training_end(model_name):
    logger.info("=" * 60)
    logger.info(f"Training Completed : {model_name}")
    logger.info("=" * 60)


# ==========================================================
# Prediction Logs
# ==========================================================

def log_prediction(model_name, prediction, confidence):
    logger.info(
        f"Model={model_name} | "
        f"Prediction={prediction} | "
        f"Confidence={confidence:.4f}"
    )


# ==========================================================
# Evaluation Logs
# ==========================================================

def log_metrics(metrics):
    logger.info("-" * 50)
    logger.info("Evaluation Metrics")

    for key, value in metrics.items():

        if isinstance(value, float):

            logger.info(f"{key:<15}: {value:.4f}")

        else:

            logger.info(f"{key:<15}: {value}")

    logger.info("-" * 50)


# ==========================================================
# Dataset Logs
# ==========================================================

def log_dataset_info(dataframe):

    logger.info("=" * 60)

    logger.info("Dataset Information")

    logger.info("=" * 60)

    logger.info(f"Rows    : {dataframe.shape[0]}")

    logger.info(f"Columns : {dataframe.shape[1]}")

    logger.info(f"Missing Values : {dataframe.isnull().sum().sum()}")

    logger.info("=" * 60)


# ==========================================================
# Model Save Logs
# ==========================================================

def log_model_saved(path):

    logger.info(f"Model Saved -> {path}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    log_info("Logger initialized successfully.")

    log_training_start("Random Forest")

    log_info("Loading dataset...")

    log_info("Preprocessing completed.")

    log_info("Training model...")

    log_training_end("Random Forest")

    log_prediction(
        "Random Forest",
        prediction=1,
        confidence=0.9623
    )

    log_warning("This is a sample warning.")

    log_error("This is a sample error.")
