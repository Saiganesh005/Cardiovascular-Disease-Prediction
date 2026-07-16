from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):

    X = df.drop("cardio", axis=1)

    y = df["cardio"]

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
