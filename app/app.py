import pickle
from fraud_model import MyModel
from predict import mongo_data, segment
from eda import clean_data
from flask import Flask, request
from flask import render_template, flash, redirect, url_for
app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", title="Home")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html", title="Home")

@app.route("/predict", methods=["POST"] )
def predict():
    df = mongo_data()
    df.pop("_id")
    df1 = clean_data(df)
    X = df1.values
    with open("model.pickle", "rb") as f:
        model = pickle.load(f)

    df["probability"] = model.predict_proba_1(X)[:, 1].round(3)
    df["fraud_level"] = df["probability"].apply(segment)
    df2 = df[df["fraud_level"].isin(["low risk", "medium risk", "high risk"])]
    df3 = df2[["object_id", "probability", "fraud_level"]]
    return render_template("predict.html", title="Home", pred = df3)


if __name__ == "__main__":

    app.run(host='0.0.0.0', port=8080, debug=True)