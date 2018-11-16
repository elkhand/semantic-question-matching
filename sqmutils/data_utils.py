import pandas as pd
from nltk import sent_tokenize, word_tokenize
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from keras import backend as K
import time
import matplotlib.pyplot as plt

def get_config(train_dataset_path,val_dataset_path=None, test_size=0.2, val_size=0.2, seed=7,is_debug_on=False):
    conf = {}
    conf["train_dataset_path"] = train_dataset_path
    conf["val_dataset_path"] = val_dataset_path
    conf['test_size'] = test_size
    conf['val_size'] = test_size
    conf["max_seq_len"] = 40
    conf["embedding_dimension"] = 100
    conf["batch_size"] = 5000
    conf["nb_epochs"] = 40 #300
    # fix random seed for reproducibility
    conf['seed'] = seed
    conf['is_debug_on'] = is_debug_on
    print("config\n",conf,"\n")
    np.random.seed(conf['seed'])
    return conf


def create_train_test_split(config):
    dfTrain = pd.read_csv(config['train_dataset_path'], sep='\t', encoding='utf-8')
    # print("dfTrain.head():\n",dfTrain.head(),"\n")
    #dfTrain.columns = ['text', 'label']
    df = dfTrain
    
    if config['val_dataset_path'] is not None:
        dfVal = pd.read_csv(config['val_dataset_path'], sep='\t', encoding='utf-8')
        #dfVal.columns = ['text', 'label']
        # print("dfVal.head():\n",dfVal.head(),"\n")
        df = pd.concat([dfTrain, dfVal])
        df = df.reset_index()
    
    return create_train_test_split_from_df(df, config)
       
    
def create_train_test_split_from_df(df, config):
    # Shuffle dataset
    # df = shuffle(df)
    if config['is_debug_on']:
        print("\n","Label distribution: ",df.groupby('is_duplicate').is_duplicate.count())
    train_x, val_x, train_y,  val_y = train_test_split(df[['id', 'qid1', 'qid2', 'question1', 'question2']],
                                                     df['is_duplicate'],
                                                     test_size=config['val_size'],
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
        x_seq = np.lib.pad(x_seq, ((0,config['max_seq_len'] - x_seq.shape[0]),(0,0)), 'constant')
    else:
        x_seq = []
        for i in range(config['max_seq_len']):
            x_seq.append(get_word_embedding(words[i], w2v, config))
        x_seq = np.array(x_seq)
    return x_seq    

def load_dataset(df, w2v, config):
    q1_embeddings = []
    q2_embeddings = []
    second_questions = []
    labels = []
    num_of_classes = 2
    for index, row in df.iterrows():
        q1 = row['question1']
        q2 = row['question2']
        label = row['is_duplicate']
        q1_words = q1.split(" ")
        q1_embedding = get_sequence_embedding(q1_words, w2v, config)
        q1_embeddings.append(q1_embedding)
        
        q2_words = q1.split(" ")
        q2_embedding = get_sequence_embedding(q2_words, w2v, config)
        q2_embeddings.append(q2_embedding)
        
        labels.append(label)
        #break
        
    df_q1_emb = np.array(q1_embeddings)
    df_q2_emb = np.array(q2_embeddings)
    df_label = np.array(labels)
    return (df_q1_emb, df_q2_emb, df_label)

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
        best_f1 =  max(val_f1_list)
        print("best_f1", best_f1)
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