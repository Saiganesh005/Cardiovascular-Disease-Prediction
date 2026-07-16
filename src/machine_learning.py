from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    return model
