from sklearn.ensemble import RandomForestClassifier
from eda import clean_data
import pandas as pd
import pickle

rf = RandomForestClassifier(n_estimators=180,
                              max_depth=30,
                              max_features=5,
                              min_samples_split=4,
                              min_samples_leaf=2,
                              criterion="gini",
                              n_jobs=-1)



class MyModel():

    def __init__(self):
        self.model = rf

    def fit(self, X, y):
        """Fit a text classifier model.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        self.model = self.model.fit(X, y)
        return self

    def predict_proba_1(self, X):
        """Make probability predictions on new data."""
        return self.model.predict_proba(X)

    def predict(self, X):
        """Make predictions on new data."""
        return self.model.predict(X)

def get_data(datafile):
    if datafile:
        df = pd.read_json(datafile)
        df1 = clean_data(df)
        y = df1.pop("label").values
        X = df1.values
    return X, y

if __name__ == "__main__":
    # pd.set_option("display.max_columns", 500)
    # X, y = get_data("../train_data.json")
    # print(X)
    # print(X.shape)
    # model = MyModel()
    # model.fit(X, y)
    # with open("model.pkl", "wb") as f:
    #     # Write the model to a file.
    #     pickle.dump(model, f)

    with open("model.pkl", "rb") as f:
        models = pickle.load(f)
    df = pd.read_json("../test_script_examples.json")

    df = clean_data(df)

    # Create features and labels
    y = df.pop("label").values
    M = df.values
    print(models.predict_proba_1(M))