# source: https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037
import os
import signal
import subprocess
from os.path import dirname, abspath

PROJECT_ROOT_DIR = dirname(dirname(abspath(__file__)))
FLAST_SERVER_DIR= os.path.join(PROJECT_ROOT_DIR, "flask_server")
MODEL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT_DIR, "semantic_question_classifier")

# Change directory to where your Flask's app.py is present
os.chdir(FLAST_SERVER_DIR) 
tf_ic_server = ""
flask_server = ""

try:
    tensorflow_model_server_start_cmd = "tensorflow_model_server "
    tensorflow_model_server_start_cmd += "--model_base_path=" + MODEL_WEIGHTS_DIR + " "
    tensorflow_model_server_start_cmd += "--rest_api_port=9000 --model_name=Semantic_Question_Matching"
    
    tf_ic_server = subprocess.Popen([tensorflow_model_server_start_cmd],
                                    stdout=subprocess.DEVNULL,
                                    shell=True,
                                    preexec_fn=os.setsid)
    print("Started TensorFlow DamageAnalyzer server!")

    flask_server = subprocess.Popen(["export FLASK_ENV=development && flask run --host=0.0.0.0"],
                                    #stdout=subprocess.DEVNULL,
                                    shell=True,
                                    preexec_fn=os.setsid)
    print("Started Flask server!")

    while True:
        print("Type 'exit' and press 'enter' to quit: ")
        in_str = input().strip().lower()
        if in_str == 'q' or in_str == 'exit':
            print('Shutting down all servers...')
            os.killpg(os.getpgid(tf_ic_server.pid), signal.SIGTERM)
            os.killpg(os.getpgid(flask_server.pid), signal.SIGTERM)
            print('Servers successfully shutdown!')
            break
        else:
            continue
except KeyboardInterrupt:
    print('Shutting down all servers...')
    os.killpg(os.getpgid(tf_ic_server.pid), signal.SIGTERM)
    os.killpg(os.getpgid(flask_server.pid), signal.SIGTERM)
print('Servers successfully shutdown!')