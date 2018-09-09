import pandas as pd
from eda import clean_data
import pickle
import time
import requests
from bson.objectid import ObjectId
from fraud_model import MyModel

from pymongo import MongoClient
client = MongoClient()
db = client.fraud
pd.set_option("display.max_columns", 500)

def live_data(n=0):
    api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
    url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
    sequence_number = n
    response = requests.post(url, json={'api_key': api_key,
                                       'sequence_number': sequence_number})
    return response.json()


def store_data(n=0):
    for iter in range(n):
        data = live_data(iter)["data"][0]
        if db.live_records.find({"object_id": data["object_id"]}).count() == 0:
            db.live_records.insert_many(data)
            time.sleep(360)


def mongo_data():
    data = list(db.live_records.find())
    df = pd.DataFrame(data)
    return df

def segment(x):
    if x < 0.5:
        return "cleared"
    elif x < 0.65:
        return "low risk"
    elif x < 0.8:
        return "medium risk"
    else:
        return "high risk"

if __name__=="__main__":
    df = mongo_data()
    df1 = clean_data(df)
    df1.pop("_id")
    X = df1.values
    # store_data(1)
    # data = live_data()
    # X = main(data)
    #Unpickle the model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    #Make the predictions
    y_predict = model.predict(X)
    y_pred_proba = model.predict_proba_1(X)
    print(y_pred_proba)
    # print(X)
    # print(y_predict)
    # #Create a Dataframe of predicted probabilities
    # df_out = pd.DataFrame(y_predict, columns=["Prediction"])
    # print(df_out)
    # #Output prediction to a file
    # df_out.to_html("templates/prediction.html")
