# Brain Tumour Detection using Convolutional Neural Networks
This repository contains code for detecting brain tumours in MRI scans using convolutional neural networks (CNNs). The code can be used with two different datasets: a larger dataset containing MRI scans of brain tumours, and a smaller dataset containing MRI scans of brain tumours, both datasets are at 240x240 resolution.

# Requirements
To run the code in this repository, you will need to set up a Python environment with the necessary packages installed. You can create a virtual environment and install the packages listed in requirements.txt using the following commands:

python -m venv env <br>
source env/bin/activate <br>
pip install -r requirements.txt <br>

# Usage
To run the final model, navigate to the final_model folder and run the following command:

cd final_model <br>
python final_model_run.py

This will train the CNN on the larger and small dataset wiht the optimial hyperparameters and display the results.

# Datasets
The datasets used in this project are not included in this repository due to their large size. However, they can be downloaded from the following sources:

Larger dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Testing
Smaller dataset: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

# Results
The resulst for this project can be found in the varying folders:
<li> The resolution testing can be found in preprocessing/res_testing
<li> The model structure can be found in the model_creation folder.
<li> Hyperparameter optimisation results can be found in the hyperparameter_testing folder

# Credits
This project was developed by James Fryer as part of MSc Data Science at University of London.

Please feel free to use this code and results within your research.
