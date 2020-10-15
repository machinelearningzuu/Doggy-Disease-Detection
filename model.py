import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import model_from_json, load_model, Sequential, Model
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K
K.clear_session()
import logging
logging.getLogger('tensorflow').disabled = True

from variables import*
from util import*
from sklearn.metrics import confusion_matrix
'''  Use following command to run the script

                python model.py
'''

class DoggySymptom(object):
    def __init__(self):
        diseases, symtoms,encoder = get_data()
        self.X = symtoms
        self.Y = diseases
        self.encoder = encoder
        print("Input Shape : {}".format(self.X.shape))
        print("Label Shape : {}".format(self.Y.shape))
        print("No: of Output classes : {}".format(len(set(self.Y))))

    def classifier(self):

        self.model = Sequential()
        n_features = self.X.shape[1]
        num_classes = len(set(self.Y))
        self.model.add(Dense(dense1, activation='relu', input_shape=(n_features,)))
        self.model.add(Dense(dense2, activation='relu'))
        self.model.add(Dense(dense2, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(dense3, activation='relu'))
        self.model.add(Dense(dense3, activation='relu'))
        self.model.add(Dense(dense4, activation='relu'))
        self.model.add(Dense(dense4, activation='relu'))
        self.model.add(Dropout(keep_prob))
        self.model.add(Dense(num_classes, activation='softmax'))

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
        # self.plot_metrics()
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

    def predicts(self,symtoms):
        P = self.model.predict(np.array([symtoms]))
        disease_idxs = P.argsort()[::-1].squeeze()
        disease_idxs = disease_idxs[:3]
        return self.encoder.inverse_transform(disease_idxs).tolist()

    def plot_confusion_matrix(self):
        P = self.model.predict(self.X).argmax(axis=-1)
        cm_ = confusion_matrix(P, self.Y)
        plt.imshow(cm_, cmap=cm.Blues)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title('Confusion matrix ')
        plt.colorbar()
        plt.savefig(confusion_matrix_img)
        plt.show()

    def predict_precautions(self, symtoms, all_diseases, all_symtoms):
        symtoms = process_prediction_data(symtoms, all_diseases, all_symtoms)
        P = self.model.predict(np.array([symtoms]))
        label = P.argmax(axis=-1)[0]
        disease = all_diseases[label]
        precausions = get_precautions(disease)
        precausions = {'precausion'+str(i): precausion for (i,precausion) in enumerate(precausions)}    
        print(precausions, disease)
        return precausions, disease

    def run(self):
        if os.path.exists(model_weights):
            print("Loading the model !!!")
            self.load_model(model_weights)
        else:
            self.classifier()
            self.train()
        # self.plot_confusion_matrix()

# symtoms = ['Fever','Nasal Discharge','Lethargy','Swollen Lymph nodes']

# model = DoggySymptom()
# model.run()
# model.predict_precautions(symtoms, all_diseases, all_symtoms)


