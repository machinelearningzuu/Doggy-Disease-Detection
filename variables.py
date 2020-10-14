import os
data_path = 'data/symtomdata.csv'
min_samples = 1800
total_samples = 3000
seed = 42
dense1 = 1024
dense2 = 512
dense3 = 128
dense4 = 64
keep_prob = 0.3

num_classes = 13

learning_rate = 0.0001
batch_size = 715
num_epoches = 10
validation_split = 0.15

host = '0.0.0.0'
port = 5000

model_weights = 'weights/disease_prediction.h5'
model2_weights = 'weights/disease_prediction2.h5'
acc_img = "visualization/accuracy_comparison.png"
loss_img = "visualization/loss_comparison.png"
confusion_matrix_img = "visualization/confusion_matrix.png"

# Model 2
n_symtoms = 20
n_diseases = 10
n_samples = 5000
stmtoms =  ['Itchin_and_redness_and_discharge_from_ear',
            'swelling_of_ears',
            'hearing_loss',
            'bleeding_from_ear',
            'Discharge_redness_or_swelling_eyes',
            'bulging_of_eyes',
            'bleeding_from_eyes', 
            'blindness_of_eyes',
            'clouding_in_eyes', 
            'yellowing_of_eyes', 
            'abnormal_gum_color']

diseases = ['kennel cough', 
            'ulcer', 
            'back problems', 
            'Mouth Cancer', 
            'paralysis', 
            'pneumonia', 
            'parasites', 
            'epilepsy', 
            'heartworm disease', 
            'Cyst']