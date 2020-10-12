import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] =' 2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import logging
logging.getLogger('tensorflow').disabled = True

K.clear_session()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from variables import*
from util import*
from sklearn.metrics import confusion_matrix
'''  Use following command to run the script
                python model.py
'''

class DoggySymptom(object):
    def __init__(self):
        diseases, symptoms = get_data2()
        self.X = symptoms
        self.Y = diseases
        print("Input Shape : {}".format(self.X.shape))
        print("Label Shape : {}".format(self.Y.shape))
        print("No: of Output classes : {}".format(len(set(self.Y))))

    def classifier(self):
        n_features = self.X.shape[1]
        num_classes = len(set(self.Y))
        inputs = Input(shape=(n_features,))
        x = Dense(dense1, activation='relu')(inputs)
        x = Dense(dense2, activation='relu')(x)
        x = Dense(dense2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dense(dense4, activation='relu')(x)
        x = Dense(dense4, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)

    def train(self):
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate),
            metrics=['accuracy'],
        )
        self.history = self.model.fit(
                            self.X,
                            self.Y,
                            batch_size=batch_size,
                            epochs=num_epoches,
                            validation_split=validation_split
                            )
        # self.save_model()

    def load_model(self, model_weights):
        K.clear_session()
        loaded_model = load_model(model_weights)
        loaded_model.compile(
                        loss='sparse_categorical_crossentropy',
                        optimizer=Adam(learning_rate),
                        metrics=['accuracy'],
                        )
        self.model = loaded_model

    def save_model(self):
        self.model.save(model2_weights)

    def predicts(self,symtoms):
        P = self.model.predict(np.array([symtoms]))
        disease_idxs = P.argsort()[::-1].squeeze()
        disease_idxs = disease_idxs[:3]
        return self.encoder.inverse_transform(disease_idxs).tolist()

    def run(self):
        if os.path.exists(model2_weights):
            print("Loading the model !!!")
            self.load_model(model2_weights)
        else:
            print("Training the model !!!")
            self.classifier()
            self.train()

if __name__ == "__main__":
    model = DoggySymptom()
    model.run()