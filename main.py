"""
==========================================================
Project : Cardiovascular Disease Prediction
File    : main.py
Author  : Robbi Sai Ganesh Devi Prasad

Description
-----------
Main entry point for the Cardiovascular Disease Prediction
project.

Options
-------
1. Train Machine Learning & Deep Learning Models
2. Predict Cardiovascular Disease
3. Exit
==========================================================
"""

import warnings

warnings.filterwarnings("ignore")

from train import main as train_models
from predict import main as predict_disease


# ==========================================================
# Banner
# ==========================================================

def banner():

    print("\n")

    print("=" * 70)
    print("      CARDIOVASCULAR DISEASE PREDICTION SYSTEM")
    print("=" * 70)

    print("Artificial Intelligence & Machine Learning Project")

    print("=" * 70)


# ==========================================================
# Menu
# ==========================================================

def menu():

    print("\n")

    print("1. Train Models")

    print("2. Predict Heart Disease")

    print("3. Exit")

    print()


# ==========================================================
# Main Application
# ==========================================================

def main():

    while True:

        banner()

        menu()

        choice = input("Enter your choice (1-3): ")

        if choice == "1":

            print("\nStarting Training Pipeline...\n")

            train_models()

            input("\nPress Enter to return to menu...")

        elif choice == "2":

            print("\nStarting Prediction System...\n")

            predict_disease()

            input("\nPress Enter to return to menu...")

        elif choice == "3":

            print("\nThank you for using the Cardiovascular Disease Prediction System.")

            print("Goodbye!\n")

            break

        else:

            print("\nInvalid choice!")

            print("Please enter 1, 2 or 3.\n")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()
