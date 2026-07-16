"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : data_loader.py
Author  : Robbi Sai Ganesh Devi Prasad

Description:
------------
Loads the cardiovascular disease dataset and performs
basic validation before returning a pandas DataFrame.
==========================================================
"""

from pathlib import Path
import pandas as pd

from src.config import DATASET_FILE


def load_data(file_path: Path = DATASET_FILE) -> pd.DataFrame:
    """
    Load the cardiovascular disease dataset.

    Parameters
    ----------
    file_path : Path
        Path to the dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If dataset file is not found.
    ValueError
        If dataset is empty.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nDataset not found!\nExpected location:\n{file_path}"
        )

    try:
        # Try semicolon-separated format (Cardio Dataset)
        df = pd.read_csv(file_path, sep=";")
    except Exception:
        # Otherwise read as a normal CSV
        df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)
    print(f"Location : {file_path}")
    print(f"Rows     : {df.shape[0]}")
    print(f"Columns  : {df.shape[1]}")
    print("=" * 60)

    return df


def dataset_info(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    """

    print("\nFirst Five Rows")
    print("-" * 60)
    print(df.head())

    print("\nDataset Shape")
    print("-" * 60)
    print(df.shape)

    print("\nColumn Names")
    print("-" * 60)
    print(df.columns.tolist())

    print("\nMissing Values")
    print("-" * 60)
    print(df.isnull().sum())

    print("\nDuplicate Records")
    print("-" * 60)
    print(df.duplicated().sum())

    print("\nStatistical Summary")
    print("-" * 60)
    print(df.describe())


def get_features_target(df: pd.DataFrame, target_column: str):
    """
    Split dataset into features and target.

    Parameters
    ----------
    df : pandas.DataFrame
    target_column : str

    Returns
    -------
    X : pandas.DataFrame
    y : pandas.Series
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


if __name__ == "__main__":

    dataframe = load_data()

    dataset_info(dataframe)
