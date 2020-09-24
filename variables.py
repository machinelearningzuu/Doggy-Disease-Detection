import os
data_path = 'data/DiseasePredictorA.csv'
min_samples = 20

dense1 = 1024
dense2 = 512
dense3 = 128
dense4 = 64
keep_prob = 0.3

num_classes = 13

learning_rate = 0.0001
batch_size = 15
num_epoches = 85
validation_split = 0.15

host = '0.0.0.0'
port = 5000

model_weights = 'weights/disease_prediction.h5'
acc_img = "visualization/accuracy_comparison.png"
loss_img = "visualization/loss_comparison.png"