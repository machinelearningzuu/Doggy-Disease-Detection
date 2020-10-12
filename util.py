import numpy as np
import pandas as pd
from variables import*
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from collections import Counter

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

def get_data2():
    symptoms = np.random.randint(2, size=(n_samples, n_symtoms))
    diseases = np.random.randint(n_diseases, size=(n_samples))

    return diseases, symptoms