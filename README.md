# Brain Tumour Detection using Convolutional Neural Networks
This repository contains code for detecting brain tumours in MRI scans using convolutional neural networks (CNNs). The code can be used with two different datasets: a larger dataset containing MRI scans of brain tumours, and a smaller dataset containing MRI scans of brain tumours, both datasets are at 240x240 resolution.

# Requirements
To run the code in this repository, you will need to set up a Python environment with the necessary packages installed. You can create a virtual environment and install the packages listed in requirements.txt using the following commands:

## Using pip and venv
python -m venv env <br>
source env/bin/activate <br>
pip install -r requirements.txt <br>

## Using Conda
conda create --name <env_name> --file requirements.txt

# Usage

## Final Model
To run the final model, navigate to the final_model folder and run the following command:

cd final_model <br>
python final_model_run.py

This will train the CNN on the larger and small dataset with the optimial hyperparameters and display the results.

## Hyperparameter optimisation
In order to run the hyperparmeter optimiation grid search for your model, follow these instructions:

1. Ensure your dataset has been properly pre-processed and tested within the hyper_model.py file <br>
2. Use the existing search space or create your own within the hyper_main.py file <br> 
3. Uncomment the configure_combinations() and hyperparameter_optimisation() <br>
4. Run the file or use 'python hyper_main.py' in the terminal <br>
5. WARNING: With the current search space this process is likely to take a matter of days, plan accordingly. <br>

# Datasets
The datasets used in this project can be downloaded from the following sources, or the pre-processed datasets can be found in the Datasets_cleaned directory:

Larger dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Testing
Smaller dataset: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

# Results
The resulst for this project can be found in the varying folders:
<li> The resolution testing can be found in preprocessing/res_testing
<li> The model structure can be found in the model_creation folder.
<li> Hyperparameter optimisation results can be found in the hyperparameter_testing folder

# Credits
This project was developed by James Fryer as part of the Final Project for MSc Data Science at University of London.

Please feel free to use this code and results within your research.
