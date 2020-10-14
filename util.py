import numpy as np
import pandas as pd
from variables import*
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from collections import Counter
import math

np.random.seed(seed)

def get_data():
    df = pd.read_csv(data_path, error_bad_lines=False)
    df_cols = df.columns.values
    df[df_cols[0]] = df[df_cols[0]].str.lower()

    Y =  df[df_cols[0]].values
    X =  df[df_cols[1:]].values

    # all_diseases = list(set(Y))
    # all_symtoms = []
    # for i in range(X.shape[0]):
    #     X_ = X[i,:].tolist()
    #     all_symtoms.extend([x for x in X_ if type(x) == str])
    # all_symtoms = list(set(all_symtoms))

    symtoms = np.zeros((len(Y), len(all_symtoms)))
    for i in range(X.shape[0]):
        X_ = X[i,:].tolist()
        for j in range(len(X_)):
            x = X_[j]
            if type(x) == str:
               x = x.lower()
               idx = all_symtoms.index(x)
               symtoms[i,idx] = 1

    symtoms, diseases = shuffle(symtoms, Y)

    all_idxs = np.random.choice(len(diseases), total_samples, replace=False)
        
    symtoms = symtoms[all_idxs]
    diseases = diseases[all_idxs]

    encoder = LabelEncoder()
    encoder.fit(diseases)
    diseases = encoder.transform(diseases)

    return diseases, symtoms, encoder

def process_prediction_data(X, all_diseases, all_symtoms):
    symtoms = np.zeros(len(all_symtoms))
    for j in range(len(X)):
        x = X[j]
        if type(x) == str:
            x = x.lower()
            idx = all_symtoms.index(x)
            symtoms[idx] = 1
    return symtoms

def get_precautions(disease):
    disease = disease.lower()
    df = pd.read_csv(precausion_path)
    df_cols = df.columns.values
    df[df_cols[0]] = df[df_cols[0]].str.lower()
    row = df.loc[df[df_cols[0]] == disease].values[1:]
    return row
