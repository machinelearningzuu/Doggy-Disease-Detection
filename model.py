import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import time
import pathlib
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
        # Multi layer perceptron model
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

        self.model = Model(
                        inputs,
                        outputs
                        )

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

    def save_model(self):  # Saving the trained model
        print("Saving the model !!!")
        self.model.save(model_weights)

    def TFconverter(self): # For deployment in the mobile devices quantization of the model using tensorflow lite
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_weights)
        converter.target_spec.supported_ops = [
                                tf.lite.OpsSet.TFLITE_BUILTINS,   # Handling unsupported tensorflow Ops 
                                tf.lite.OpsSet.SELECT_TF_OPS 
                                ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]      # Set optimization default and it configure between latency, accuracy and model size
        tflite_model = converter.convert()

        model_converter_file = pathlib.Path(model_converter) 
        model_converter_file.write_bytes(tflite_model) # save the tflite model in byte format
 
    def TFinterpreter(self):
        self.interpreter = tf.lite.Interpreter(model_path=model_converter) # Load tflite model
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details() # Get input details of the model
        self.output_details = self.interpreter.get_output_details() # Get output details of the model

    def Inference(self, symtoms):
        symtoms = symtoms.astype(np.float32)
        input_shape = self.input_details[0]['shape']
        assert np.array_equal(input_shape, symtoms.shape), "Input tensor hasn't correct dimension"

        self.interpreter.set_tensor(self.input_details[0]['index'], symtoms)

        self.interpreter.invoke() # set the inference

        output_data = self.interpreter.get_tensor(self.output_details[0]['index']) # Get predictions
        return output_data

    def run(self):
        if not os.path.exists(model_converter):
            if not os.path.exists(model_weights):
                self.classifier()
                self.train()

            self.TFconverter()
        self.TFinterpreter()    

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
        symtoms = symtoms.reshape(1,-1)
        P = self.Inference(symtoms)
        label = P.argmax(axis=-1)[0]
        disease = all_diseases[label]
        precausions = get_precautions(disease)
        precausions = {'precausion'+str(i): precausion for (i,precausion) in enumerate(precausions)}    
        return precausions, disease

# symtoms = ['Fever','Nasal Discharge','Lethargy','Swollen Lymph nodes']

# model = DoggySymptom()
# model.run()
# print(model.predict_precautions(symtoms, all_diseases, all_symtoms))


