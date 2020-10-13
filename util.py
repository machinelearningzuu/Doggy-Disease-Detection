import numpy as np
import pandas as pd
from variables import*
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from collections import Counter
import math

def get_data():
    df = pd.read_csv(data_path)
    df_cols = df.columns.values
    df[df_cols[-1]] = df[df_cols[-1]].str.lower()

    diseases =  df[df_cols[-1]].values
    symptoms =  df[df_cols[:-1]].values

    sample_count = dict(Counter(diseases))
    relevant_diseases = [k for k,v in sample_count.items() if v > min_samples]
    data = df.loc[df[df_cols[-1]].isin(relevant_diseases)]
    diseases =  data[df_cols[-1]].values
    symptoms =  data[df_cols[:-1]].values

    diseases, symptoms = shuffle(diseases, symptoms)
    encoder = LabelEncoder()
    encoder.fit(diseases)
    diseases = encoder.transform(diseases)

    scalar = StandardScaler()
    scalar.fit(symptoms)
    # symptoms = scalar.transform(symptoms)

    return diseases, symptoms, encoder

def get_chatbot_data():
    df = pd.read_csv(chatbot_data_path)
    df_cols = df.columns.values
    df[df_cols[-1]] = df[df_cols[-1]].str.lower()

    Y = df[df_cols[0]].values
    X = df[df_cols[1:]].values
    all_symtoms = []
    for i in range(X.shape[0]):
        X_ = X[i,:].tolist()
        all_symtoms.extend([x for x in X_ if type(x) == str])
    all_symtoms = list(set(all_symtoms))

    symtoms = np.zeros((len(Y), len(all_symtoms)))
    for i in range(X.shape[0]):
        X_ = X[i,:].tolist()
        for j in range(len(X_)):
            x = X_[j]
            if type(x) == str:
               idx = all_symtoms.index(x)
               symtoms[i,idx] = 1

    symtoms, diseases = shuffle(symtoms, Y)

    encoder = LabelEncoder()
    encoder.fit(diseases)
    diseases = encoder.transform(diseases)

    return diseases, symtoms, encoder
