# source: https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037
# importing the requests library
import argparse
import requests
import sys
import json


API_ENDPOINT = "http://localhost:5000/SQClassifier/predict/"

ap = argparse.ArgumentParser()
ap.add_argument("-q1", "--question1", required=True,
                help="Question 1")
ap.add_argument("-q2", "--question2", required=True,
                help="Question 2")
args = vars(ap.parse_args())

# defining the api-endpoint
print("args", args)

q1_eng = args["question1"]
q2_eng = args["question2"]


# data to be sent to api
data = {'q1': q1_eng, "q2" : q2_eng}

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, data=data)

# extracting the response
print("{}".format(r.text))

result = json.loads(r.content.decode('utf-8'))
print("result", result['predictions'][0][0])