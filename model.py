import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K
import logging
logging.getLogger('tensorflow').disabled = True

from variables import*
from util import*

'''  Use following command to run the script
                python model.py
'''

class DoggySymptom(object):
    def __init__(self):
        diseases, symptoms, encoder = get_data()
        self.X = symptoms
        self.Y = diseases
        self.encoder = encoder
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
        self.plot_metrics()
        self.save_model()

    def plot_metrics(self):
        loss_train = self.history.history['loss']
        loss_val = self.history.history['val_loss']
        plt.plot(np.arange(1,num_epoches+1), loss_train, 'r', label='Training loss')
        plt.plot(np.arange(1,num_epoches+1), loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(loss_img)
        plt.legend()
        plt.show()

        acc_train = self.history.history['accuracy']
        acc_val = self.history.history['val_accuracy']
        plt.plot(np.arange(1,num_epoches+1), acc_train, 'r', label='Training Accuracy')
        plt.plot(np.arange(1,num_epoches+1), acc_val, 'b', label='validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(acc_img)
        plt.legend()
        plt.show()

    def load_model(self, model_weights):
        loaded_model = load_model(model_weights)
        loaded_model.compile(
                        loss='sparse_categorical_crossentropy',
                        optimizer=Adam(learning_rate),
                        metrics=['accuracy'],
                        )
        self.model = loaded_model

    def save_model(self):
        self.model.save(model_weights)

    def predicts(self,X):
        return self.model.predict(X)

    def run(self):
        if os.path.exists(model_weights):
            print("Loading the model !!!")
            self.load_model(model_weights)
        else:
            print("Training the model !!!")
            self.classifier()
            self.train()

model = DoggySymptom()
model.run()