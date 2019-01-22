# inspired from: https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037
import json

import requests
from flask import Flask, request, jsonify

import os
import os.path
from os.path import dirname, abspath
import sys

PROJECT_ROOT_DIR = dirname(dirname(abspath(__file__)))

WORD_EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT_DIR,"word_embeddings") 
embedding_path = os.path.join(WORD_EMBEDDINGS_DIR, "wiki.en.vec")


sys.path.append(str(PROJECT_ROOT_DIR))
import sqmutils.data_utils as du

import time

app = Flask(__name__)

w2v = {}
config = {}

def download_word_embedding():
    import urllib.request
    start = time.time()
    url = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec"   
    urllib.request.urlretrieve(url, embedding_path)
    end = time.time()
    print("Total time for downloading word embeddings: ", (end - start))



@app.before_first_request
def do_something_only_once():
    global config,w2v
    #Load embeddings
    print("word vectors path", embedding_path)
    start = time.time()
    if not os.path.exists(embedding_path):
        print("\n === Will download Fasttext word embeddings, will take few minutes to complete ===\n")
        download_word_embedding()
        
    
    w2v = du.load_embedding(embedding_path)
    # Initialize configs
    config = du.get_config("",0,0)
    print("config", config)
    end = time.time()
    print("Total time passed: ", (end-start))


@app.route('/SQClassifier/predict/', methods=['POST'])
def semantic_question_classifier():
    row = {"question1" : request.form['q1'], "question2" : request.form['q2']}
    q1_embedding, q2_embedding = du.load_dataset_single_row(row, w2v,config)
    
    # Remove batch dimension (1,32,300) -> (32,300)
    q1_embedding = q1_embedding[0, :, :]
    q2_embedding = q2_embedding[0, :, :]
    
    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'q1': q1_embedding.tolist(), 'q2': q2_embedding.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:9000/v1/models/Semantic_Question_Matching:predict', json=payload)
    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))
    # Returning JSON response to the frontend
    return jsonify(pred)