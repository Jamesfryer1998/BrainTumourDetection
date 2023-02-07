import sys
import json
from hyper_model import MultiClassModelCreation
sys.path.append('/Users/james/MScCode/Final Project/')
from preprocessing.emailer import email

root_path = '/Users/james/MScCode/Final Project/Datasets_cleaned/240_resolution/brain_tumour_large'

# Initialising model
model_creation = MultiClassModelCreation(root_path)
# # Processing data
model_creation.process_data()

def configure_combinations():
    conv_1_2_units = [32, 64, 128, 256]
    conv_3_4_units = [32, 64, 128, 256]
    dense_units = [64, 128, 256, 512]
    epochs = [10, 15, 25, 50]

    combinations = []

    for conv_1_2_unit in conv_1_2_units:
        for conv_3_4_unit in conv_3_4_units:
            for dense_unit in dense_units:
                for epoch in epochs:
                    combinations.append([conv_1_2_unit, conv_3_4_unit, dense_unit, epoch])

    with open("hyperparameter_testing/combinations.json", "w") as outfile:
        json.dump(combinations, outfile, indent=1)

# Only use this for the first run
# configure_combinations()

with open("hyperparameter_testing/combinations.json", "r") as infile:
    data = json.load(infile)

for combination in data:
    model_creation.build_model(combination[0], combination[1], combination[2], combination[3])
    try:
        email(type='hyper_testing', combination=combination)
    except:
        continue
    
    print(f'Combination complete: {combination}')