#!/bin/bash
clear

EMBEDDINGS_PATH="../word_embeddings"
DATASET_PATH="../dataset"

if [ ! -d "$EMBEDDINGS_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir  -p -- "$EMBEDDINGS_PATH"
fi

if [ ! -d "$DATASET_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir  -p -- "$DATASET_PATH"
fi

url="https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec"   
local_embedding_path="$EMBEDDINGS_PATH/wiki.en.vec"
echo "Downloading Fasttext word embeddings from " $url " to " $local_embedding_path
echo `curl -# -C - -o $local_embedding_path $url`


DATASET_URL="http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
local_dataset_path="$DATASET_PATH/quora_duplicate_questions.tsv"
echo "Downloading Quora Questiosn dataset " $DATASET_URL " to " $local_dataset_path
echo `curl -# -C - -o $local_dataset_path $DATASET_URL`


TEST_DATASET_URL="https://storage.googleapis.com/kaggle-competitions-data/kaggle/6277/test.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1548306855&Signature=V4S3wmWK%2BHNAVQLcFtlfNVly1tcxpenTzrIwzAYfxXSUeaOtmEbxXS2H2An8mlSo8zaaS1uLdWOlqH9SmDDW6RUj9gQUz%2FQ9rXlZeYU7oymGCuZ6JvPOUsSqk1vCqusybo4RYnX3mgDSt61lY%2BSmYJnasBV26bgvv%2BJuh%2BV9F29ekzr013bTHP7wUmwKWEgridJoFdb03yrXmS%2Fv9LpnFjxU6F%2FZ20wZ2VTekXNBzl7HLJ5Q4VKqVTcv%2Bagx0HwThSXfQKUNBINSYxW1jyen7pidLpcR2B9X4n43jc9kzZQQRK5IJ2NZbPzWpsLB05g5%2F7aQ%2FDMxCjjQ3RCTj5XKmQ%3D%3D"
# https://www.kaggle.com/c/quora-question-pairs/data
local_test_dataset_path="$DATASET_PATH/test_quora_duplicate_questions.csv.zip"
echo "Downloading Test Quora Questiosn dataset " $TEST_DATASET_URL " to " $local_test_dataset_path
echo `curl -# -C - -o $local_test_dataset_path $TEST_DATASET_URL`
cd $DATASET_PATH
unzip $local_test_dataset_path
rm $local_test_dataset_path

