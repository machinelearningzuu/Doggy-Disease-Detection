import os
data_path = 'DiseasePredictor.csv'
min_samples = 5

dense1 = 1024
dense2 = 512
dense3 = 128
keep_prob = 0.3

num_classes = 13

learning_rate = 0.0001
batch_size = 10
num_epoches = 100
validation_split = 0.1

model_weights = 'disease_prediction.h5'
acc_img = "accuracy_comparison.png"
loss_img = "loss_comparison.png"