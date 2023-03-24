# Brain Tumour Detection using Convolutional Neural Networks
This repository contains code for detecting brain tumours in MRI scans using convolutional neural networks (CNNs). The code can be used with two different datasets: a larger dataset containing MRI scans of brain tumours at 240 resolution, and a smaller dataset containing MRI scans of brain tumours at 144 resolution.

# Requirements
To run the code in this repository, you will need to set up a Python environment with the necessary packages installed. You can create a virtual environment and install the packages listed in requirements.txt using the following commands:

bash
Copy code
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Usage
To run the final model, navigate to the final_model folder and run the following command:

bash
Copy code
python run_model.py
This will train the CNN on the larger dataset and save the results in the results folder.

# Datasets
The datasets used in this project are not included in this repository due to their large size. However, they can be downloaded from the following sources:

Larger dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Testing
Smaller dataset: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

# Results
The results of the final model can be found in the results folder. The accuracy.png and loss.png files show the accuracy and loss of the model over the training epochs, and the confusion_matrix.png file shows the confusion matrix of the model's predictions.

# Credits
This project was developed by James Fryer as part of MSc Data Science at University of London.

Please feel free to use this code and results within your research.
