"""A Python Page Summarizer API"""

import pickle
from flask import Flask, request, jsonify
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
    """ Returns a message to user """
    return {"message": "This is SMS spam detection model.format{'message':'SMS message'} and POST."}


@APP.route('/', methods=['POST'])
def predict():
    """ get data and return response """
    data = request.get_json(force=True)
    data = data['message']
    data = [data]
    arrt = COUNTVEC.transform(data).toarray()

    my_prediction = CLF.predict(arrt)

    if my_prediction[0] == 0:
        response = "Not a spam"
    elif my_prediction[0] == 1:
        response = "This is a spam o!"

    return jsonify(results=response)


if __name__ == '__main__':
    APP.run(port=5000, debug=True)
