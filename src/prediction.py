import joblib

model = joblib.load("models/random_forest.pkl")

def predict(data):

    return model.predict(data)
