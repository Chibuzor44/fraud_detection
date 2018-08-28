import pandas as pd
from eda import clean_data
import pickle
from fraud_model import MyModel

#Loading input
df = pd.read_json("../test_script_examples.json")

# import requests
# api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
# url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
# sequence_number = 0
# response = requests.post(url, json={'api_key': api_key,
#                                    'sequence_number': sequence_number})
# raw_data = response.json()
#
# df = pd.DataFrame(raw_data)
# print(df)


#clean the data
df = clean_data(df)

#Create features and labels
y = df.pop("label").values
X = df.values

#Unpickle the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


#Make the predictions
y_predict = model.predict(X)
y_pred_proba = model.predict_proba_1(X)

#Create a Dataframe of predicted probabilities
df_out = pd.DataFrame(y_predict, columns=["Prediction"])
print(df_out)
#Output prediction to a file
df_out.to_html("templates/prediction.html")
