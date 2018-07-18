import pandas as pd
from nltk import sent_tokenize, word_tokenize
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle

def get_config(train_dataset_path,val_dataset_path=None, test_size=0.2, seed=7,is_debug_on=False):
    conf = {}
    conf["train_dataset_path"] = train_dataset_path
    conf["val_dataset_path"] = val_dataset_path
    conf['test_size'] = test_size
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
                                                     test_size=config['test_size'],
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