import os
from os.path import dirname, abspath
import time
import argparse
import requests
import sys
import json
import numpy as np

import pandas as pd

API_ENDPOINT = "http://localhost:5000/SQClassifier/predict/"

PROJECT_ROOT_DIR = dirname(dirname(abspath(__file__)))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT_DIR,"dataset","test.csv")
test_probabilities_csv = os.path.join(PROJECT_ROOT_DIR,"dataset","test_results.csv")


def write_to_file(lines, csv_file):
	for line in lines:
		csv_file.write(line)


start = time.time()
dfTest = pd.read_csv(TEST_DATA_DIR, sep=',', encoding='utf-8')
valid_ids =[type(x)==int for x in dfTest.test_id] 
dfTest = dfTest[valid_ids].drop_duplicates()
dfTest = dfTest.replace(np.nan, '', regex=True)
print("Total test examples", len(dfTest))
end = time.time()
print("Total time passed", (end - start))

start = time.time()
lines = []
with open(test_probabilities_csv, "w") as csv_file:
	for index, row in dfTest.iterrows():
		data = {'q1': row["question1"], "q2" : row["question2"]}
		if row['test_id'] % 1000 == 0:
			print(row['test_id'])

		test_id = str(row['test_id'])
		# sending post request and saving response as response object
		r = requests.post(url=API_ENDPOINT, data=data)
		result = json.loads(r.content.decode('utf-8'))
		prob = round(result['predictions'][0][0],1)
		#line = str(index) + ",\"" + row["question1"] + "\",\"" + row["question2"] + "\"," + str(result['predictions'][0][0]) + "\n"
		line = test_id + "," + str(prob) + "\n"
		lines.append(line)
		if len(lines) == 10000:
			write_to_file(lines, csv_file)
			lines = []
		
end = time.time()
print("Total time passed", (end - start))


