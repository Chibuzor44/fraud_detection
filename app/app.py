import pickle
import pandas as pd
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

    df["probability of fraud"] = model.predict_proba_1(X)[:, 1].round(3)
    df["risk_level"] = df["probability of fraud"].apply(segment)
    df2 = df[df["risk_level"].isin(["low risk", "medium risk", "high risk"])]
    df2.to_html("templates/prediction.html")
    df2.to_csv("prediction.csv")
    return render_template("prediction.html", title="Home")


if __name__ == "__main__":
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    app.run(host='0.0.0.0', port=8084, debug=True)