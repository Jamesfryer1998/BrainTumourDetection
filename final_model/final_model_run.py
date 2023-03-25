import sys
sys.path.append('/Users/james/MScCode/Final Project/')
from hyperparameter_testing.hyper_model import MultiClassModelCreation
from hyperparameter_testing.small_dataset_model import TestSmallDataset
from hyperparameter_testing.results_evaluation import evaluate_hyperparameter_optimisation

conv_1_2, con_3_4, dense, epoch = evaluate_hyperparameter_optimisation(print_results=True)

def larger_dataset_model():
    root_path = 'Datasets_cleaned/240_resolution/brain_tumour_large'

    # Initialising model
    large_model = MultiClassModelCreation(root_path)

    # Processing data
    large_model.process_data()

    # This model is based off the optimial parameters found from the results_evaluation of the hyperparameter optimisation
    large_model.build_model(conv_1_2, con_3_4, dense, epoch, save_results=False)

def smaller_dataset_model():

    root_path = 'Datasets_cleaned/240_resolution/brain_tumour_small'

    # Initialising model
    small_model = TestSmallDataset(root_path)

    # Processing data
    small_model.process_data()

    # This model is based off the optimial parameters found from the results_evaluation of the hyperparameter optimisation
    small_model.build_model(conv_1_2, con_3_4, dense, 100, save_results=False)


larger_dataset_model()
smaller_dataset_model()
