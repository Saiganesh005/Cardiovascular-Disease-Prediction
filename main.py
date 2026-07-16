from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.machine_learning import train_models

def main():

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    train_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
