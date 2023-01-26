import sys
from model_testing import MultiClassModelCreation

sys.path.append('/Users/james/MScCode/Final Project/')
from preprocessing.emailer import email

root_path = '/Users/james/MScCode/Final Project/Datasets_cleaned/240_resolution/brain_tumour_large'
conv_layers = [1,2,3,4]
dense_layers = [1,2,3,4]

# Initialising model
model_creation = MultiClassModelCreation(root_path)
# # Processing data
model_creation.process_data()

for conv in conv_layers:
    for dense in dense_layers:
        model_creation.build_model(conv, dense)
        email(conv=conv, dense=dense)
        


