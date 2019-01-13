import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
import keras
from keras import backend as K
import time
import matplotlib.pyplot as plt
import re
import os


def get_config(train_dataset_path, test_size, val_size, seed=7, embedding_dimension=100, is_debug_on=False):
    conf = {}
    conf["train_dataset_path"] = train_dataset_path
    conf['test_size'] = test_size
    conf['val_size'] = val_size
    conf["max_seq_len"] = 32 #40
    conf["embedding_dimension"] = embedding_dimension
    conf["batch_size"] = 3096
    conf["nb_epochs"] = 100 #300
    conf["recurrent_dropout"] = 0.3
    conf["dropout"] = 0.3
    # fix random seed for reproducibility
    conf['seed'] = seed
    conf['is_debug_on'] = is_debug_on
    print("config\n",conf,"\n")
    np.random.seed(conf['seed'])
    return conf


def create_train_test_split(config):
    dfTrain = pd.read_csv(config['train_dataset_path'], sep='\t', encoding='utf-8')
    
    # Clean the questions
    dfTrain['question1'] = dfTrain['question1'].apply(clean)
    dfTrain['question2'] = dfTrain['question2'].apply(clean)

    return create_train_test_split_from_df(dfTrain, config)
       
    
def create_train_test_split_from_df(df, config, isValSplit=False):
    # Shuffle dataset
    # df = shuffle(df)
    if config['is_debug_on']:
        print("\n","Label distribution: ",df.groupby('is_duplicate').is_duplicate.count())
    
    testOrValSplitRatio = config['val_size'] if isValSplit else config['test_size']
    train_x, val_x, train_y,  val_y = train_test_split(df[['question1', 'question2']], # 'id', 'qid1', 'qid2',
                                                     df['is_duplicate'],
                                                     test_size=testOrValSplitRatio,
                                                     random_state=config['seed'],
                                                     stratify=df["is_duplicate"])

    trainDataset = pd.concat([train_x, train_y], axis=1)
    valDataset = pd.concat([val_x, val_y], axis=1)

    trainDataset = trainDataset.dropna()
    trainDataset = trainDataset.reset_index(drop=True)
    valDataset = valDataset.dropna()
    valDataset = valDataset.reset_index(drop=True)
    if config['is_debug_on']:
        print("\n","trainDataset Label distribution: ",trainDataset.groupby('is_duplicate').is_duplicate.count(), "\n")
        print("\n","valDataset Label distribution: ",valDataset.groupby('is_duplicate').is_duplicate.count() , "\n")
        print("trainDataset.head():\n", trainDataset.head(),"\n")
        print("valDataset.head():\n", valDataset.head(),"\n")
    return [trainDataset, valDataset]


def save_to_file(df, file_name):
    createDir(Path(file_name).parent)
    df.to_csv(file_name, header=None, index=False, encoding='utf-8')

def createDir(outputDir):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

def write_to_file(labelsSet, filename):
    with open(filename, 'w') as f:
        cnt = 0 
        for label in labelsSet:
            if cnt != len(labelsSet)-1:
                f.write(label + "\n")
            else:
                f.write(label)
            cnt += 1
        

def load_embedding(path):
    w2v = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            entries = line.rstrip().split(" ")
            word, entries = entries[0], entries[1:]
            w2v[word] = np.array(entries).astype(np.float) # Convert String type to float
    print('embedding size : %d' % len(w2v))
    print('embedding dimension : %s' % (w2v['apple'].shape,))
    return w2v

def get_word_embedding(word, w2v, config):
    word = word.lower()
    if word in w2v:
        return w2v[word]
    else:
        return np.zeros(config['embedding_dimension'],)


def get_sequence_embedding(words, w2v, config):
    if len(words) <= config['max_seq_len']:
        # Add padding
        x_seq = np.array([get_word_embedding(word, w2v, config) for word in words])
        #x_seq = np.lib.pad(x_seq, ((0,config['max_seq_len'] - x_seq.shape[0]),(0,0)), 'constant')
    else:
        x_seq = []
        for i in range(config['max_seq_len']):
            x_seq.append(get_word_embedding(words[i], w2v, config))
        x_seq = np.array(x_seq)
    return x_seq    

def load_dataset(df, w2v, config, isTestDataset=False):
    q1_embeddings = []
    q2_embeddings = []
    second_questions = []
    labels = []
    num_of_classes = 2
    for index, row in df.iterrows():
        q1 = row['question1']
        q2 = row['question2']
        
        try:
            q1_words = nltk.word_tokenize(q1)
            q1_embedding = get_sequence_embedding(q1_words, w2v, config)
            q2_words = nltk.word_tokenize(q2)
            q2_embedding = get_sequence_embedding(q2_words, w2v, config)
        except:
            print(index, "causing error: ' ",row," '")
            continue
        
        if not isTestDataset:
            label = row['is_duplicate']
            labels.append(label)

        q1_embeddings.append(q1_embedding)
        q2_embeddings.append(q2_embedding)
        
    q1_embeddings = keras.preprocessing.sequence.pad_sequences(q1_embeddings, dtype='float32')
    q2_embeddings = keras.preprocessing.sequence.pad_sequences(q2_embeddings, dtype='float32')

    df_q1_emb = np.array(q1_embeddings)
    df_q2_emb = np.array(q2_embeddings)
    if not isTestDataset:
        df_label = np.array(labels)
    
    return (df_q1_emb, df_q2_emb, df_label) if not isTestDataset else (df_q1_emb, df_q2_emb)

def generate_model_name(filename, best_acc_val):
    timestamp = str(time.time()).split(".")[0]
    best_acc_val = round(best_acc_val,4)
    filename += "-" + str(best_acc_val) + "-" + timestamp
    return filename

def plot_model_accuracy(history, modelDir="", hasF1=False):
    """plot acc and loss for train and val"""
    base_filename = "semantic" 
    filename = generate_model_name(base_filename + "-acc", max(history.history['val_acc']))
    fig = plt.figure()
    val_acc_list = history.history['val_acc']
    best_val_acc =  max(val_acc_list)
    print("best_train_acc", max(history.history['acc']))
    print("best_val_acc", best_val_acc)
    
    if hasF1:
        val_f1_list = history.history['val_f1']
        best_val_f1 =  max(val_f1_list)
        print("best_val_f1", best_val_f1)
        print("best_train_f1", max(history.history['f1']))
    
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    fig.savefig(modelDir + "/" + filename + ".png") 
    
    # "Loss"
    fig = plt.figure()
    filename = generate_model_name(base_filename + "-loss", min(history.history['val_loss']))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    fig.savefig(modelDir + "/" + filename + ".png")  

    if hasF1:
        fig = plt.figure()
        filename = generate_model_name(base_filename + "-f1", max(history.history['val_f1']))
        plt.plot(history.history['f1'])
        plt.plot(history.history['val_f1'])
        plt.title('model F1')
        plt.ylabel('F1')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        fig.savefig(modelDir + "/" + filename + ".png")  


def precision(y_true, y_pred):
    """source: https://github.com/keras-team/keras/issues/5400"""
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """source: https://github.com/keras-team/keras/issues/5400"""
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    """source: https://github.com/keras-team/keras/issues/5400"""
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+ K.epsilon()))

def clean(input):
    # source: https://www.kaggle.com/moseli/clean-questions
    #input = input.lower()
    return re.sub('[!@#.,/$%^&*\(\)\{\}\[\]-_\<\>?\'\";:~`]',' ',str(input))

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)