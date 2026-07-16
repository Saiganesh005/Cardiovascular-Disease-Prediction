from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_ann(input_dim):

    model = Sequential()

    model.add(Dense(64, activation="relu", input_shape=(input_dim,)))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    return model
