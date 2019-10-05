import pickle
from flask import Flask, render_template, url_for, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# load model
COUNTVEC = pickle.load(open("models/Countvec.pkl", "rb"))
SPAMMODEL = open('models/spammodel.pkl', 'rb')
CLF = joblib.load(SPAMMODEL)

# app
APP = Flask(__name__)

# routes


@APP.route('/', methods=['GET'])
def home():
    # Give message to user
    return {"message": "This is SMS spam detection model. Use the format {'message': 'SMS message'} and POST to get prediction."}


@APP.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)
    data = data['message']
    data = [data]
    # convert data into array
    arrt = COUNTVEC.transform(data).toarray()

    # predictions
    my_prediction = CLF.predict(arrt)

    # check predicted value
    if my_prediction[0] == 0:
        response = "Not a spam"
    elif my_prediction[0] == 1:
        response = "This is a spam o!"

    # return data
    return jsonify(results=response)


if __name__ == '__main__':
    APP.run(port=5000, debug=True)
