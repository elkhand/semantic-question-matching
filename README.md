# Semantic Question Matching
Semantic Question Matching with Deep Learning Keras

This is Keras implementation of [Semantic Question Matching with Deep Learning](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning)

There was also [Kaggle competition](https://www.kaggle.com/c/quora-question-pairs/data).

## Dataset

You can download data from: http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv 
Dataset info: https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs 


## Word embedding
300 dimensional [Fasttext word embeddings](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec) are used.

## Data Cleaning
Not much data cleaning was done.
1. all words are converted into lower case
2. removing punctuations
```
def clean(input):
    input = input.lower()
    return re.sub('[!@#.,/$%^&*\(\)\{\}\[\]-_\<\>?\'\";:~`]',' ',str(input))
```

## Model : Shared Bi-LSTM model
 
`Total params`: 1,436,161

Model architecture:

<img src="models/model_architecture.png" height="400" alt="Shared Bi-LSTM model"/>


### Result on test dataset:

```
{
    "acc": 0.8609943112287962,
    "f1": 0.8099339832446966,
    "loss": 0.40213754246089234,
    "precision": 0.7910408705966826,
    "recall": 0.8450007286515238
}
```

### Train and val accuracy

<img src="models/semantic-acc-0.8651-1547439696.png" height="400" alt="Train and val accuracy"/>

### Train and val loss

<img src="models/semantic-loss-0.321-1547439696.png" height="400" alt="Train and val loss"/>

### Train and val f1 score

<img src="models/semantic-f1-0.8222-1547439696.png" height="400" alt="Train and val f1 score"/>

# Running code

You need to create `Python 3.6` environment:
```
conda env create -f environment.yml
```


Now you can run these commadns below to download the word embeddings file and dataset:
```
cd scripts
./download_files.sh
```

This will create 2 folders and will download corresponding files into those directories:
```
project_dir/word_embeddings/wiki.en.vec
proejct_dir/dataset/quora_duplicate_questions.tsv
```

You can install tensorflow-server-model as described in [this blog post](https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037):

Now you can start Flask app and tensorflow-server-model:

```
cd scripts
python start-flask-and-tensorflow_model_server.py
```

Now in anohter terminal window you can run this script for testing a question pair semantic similarity:
```
cd scripts
python flask_sample_request.py -q1="What is 2 + 3 ?" -q2="What is 2 + 3 ?"
```

**Note**: The first request response time will take around ~6 min, as the Flask app will load word embeddings into memory (which is around 6.2 GB), but all other requests will return results in milliseconds.

## Generating test results for Kaggle

You can compute probabilities for [`Quora Question Pairs`](https://www.kaggle.com/c/quora-question-pairs#evaluation) test dataset using `Evaluating-on-test-data.ipynb` notebook.

```
saved_model_cli show --dir semantic_question_classifier/1 --all

saved_model_cli run --dir semantic_question_classifier/1 --tag_set serve --signature_def serving_default --input_exp 'q1=np.random.rand(1,32, 300);q2=np.random.rand(1,32,300) '

```