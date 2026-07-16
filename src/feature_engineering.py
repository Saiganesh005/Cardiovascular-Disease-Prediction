def feature_selection(df):

    return df.drop("cardio", axis=1), df["cardio"]
