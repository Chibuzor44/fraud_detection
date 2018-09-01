import pandas as pd
from eda import clean_data
import pickle
import requests
from fraud_model import MyModel

#Loading input
# df = pd.read_json("../test_script_examples.json")

def live_data():
    api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
    url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
    sequence_number = 0
    response = requests.post(url, json={'api_key': api_key,
                                       'sequence_number': sequence_number})
    return response.json()


def main(raw_data):
    data = raw_data["data"][0]
    df = pd.DataFrame([data], columns=data.keys())
    df = clean_data(df)

    # Create features
    X = df.values
    return X


if __name__=="__main__":
    data = live_data()
    X = main(data)
    #Unpickle the model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    #Make the predictions
    y_predict = model.predict(X)
    # y_pred_proba = model.predict_proba_1(X)

    # print(X)
    print(y_predict)
    # #Create a Dataframe of predicted probabilities
    # df_out = pd.DataFrame(y_predict, columns=["Prediction"])
    # print(df_out)
    # #Output prediction to a file
    # df_out.to_html("templates/prediction.html")
