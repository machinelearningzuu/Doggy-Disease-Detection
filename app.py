import os
import json
import pandas as pd

from variables import *
from model import DoggySymptom
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True  # Prevent display tensorflow warining and memory loading

import tensorflow as tf
if tf.config.list_physical_devices('GPU'): # Check whether The running instance has GPU or not
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True) # If has set memory growth

from flask import Flask
from flask import jsonify
from flask import request
'''
        python -W ignore app.py
'''

app = Flask(__name__)
model = DoggySymptom()
model.run()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    symtoms = eval(message['symtoms'])
    precausions, disease = model.predict_precautions(symtoms, all_diseases, all_symtoms)
    response = {
            'diseases': disease,
            'precausions': precausions
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host=host, port=port, threaded=False, use_reloader=False)
