import os
data_path = 'data/symtomdata.csv'
precausion_path = 'data/precausions.csv'
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
batch_size = 600
num_epoches = 10
validation_split = 0.15

host = '0.0.0.0'
port = 5000

model_converter = "weights/model.tflite"
model_weights = 'weights/disease_prediction.h5'
acc_img = "visualization/accuracy_comparison.png"
loss_img = "visualization/loss_comparison.png"
confusion_matrix_img = "visualization/confusion_matrix.png"

# Model 2
n_symtoms = 20
n_diseases = 10
n_samples = 5000
all_symtoms =  ['Acute blindness', 'Urine infection', 'Red bumps', 'Loss of Fur', 'Licking', 'Grinning appearance', 'Coughing', 'Eye Discharge', 'Seizures', 'excess jaw tone', 'Coma', 'Weakness', 'Wounds', 'Neurological Disorders', 'blood in stools', 'Stiff and hard tail', 'Dry Skin', 'Lameness', 'Swelling of gum', 'Fever', 'Bloated Stomach', 'Face rubbing', 'Aggression', 'Wrinkled forehead', 'Lumps', 'Plaque', 'Blindness', 'Weight Loss', 'Swollen Lymph nodes', 'Excessive Salivation', 'Loss of Consciousness', 'Tender abdomen', 'Purging', 'Dandruff', 'Loss of appetite', 'Pale gums', 'Collapse', 'Constipation', 'Hunger', 'Discomfort', 'Pain', 'Paralysis', 'Red patches', 'Fur loss', 'Losing sight', 'WeightLoss', 'Sepsis', 'Increased drinking and urination', 'Bad breath', 'Itchy skin', 'Receding gum', 'Irritation', 'Enlarged Liver', 'Eating grass', 'Nasal Discharge', 'Depression', 'lethargy', 'Stiffness of muscles', 'Eating less than usual', 'Scratching', 'Severe Dehydration', 'Tartar', 'Cataracts', 'Swelling', 'Redness of gum', 'Diarrhea', 'Scabs', 'Breathing Difficulty', 'Difficulty Urinating', 'Continuously erect and stiff ears', 'Glucose in urine', 'Burping', 'Passing gases', 'Vomiting', 'Blood in urine', 'Smelly', 'Redness around Eye area', 'Bleeding of gum', 'Bloody discharge', 'Redness of skin', 'Lethargy', 'Abdominal pain', 'Lack of energy', 'Anorexia', 'Heart Complication', 'Yellow gums']
all_diseases = ['hepatitis ', 'diabetes', 'cancers', 'allergies', 'tetanus ', 'gingitivis', 'skin rashes', 'distemper', 'parvovirus', 'chronic kidney disease ', 'tick fever', 'gastrointestinal disease']
all_symtoms = list(map(str.lower,all_symtoms))
all_diseases = list(map(str.lower,all_diseases))
