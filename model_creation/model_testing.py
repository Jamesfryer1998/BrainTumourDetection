import os
import json
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tqdm
import cv2
import datetime

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class MultiClassModelCreation:
    def __init__(self, root_path):
        self.root_path = root_path

    def process_data(self):
        t1 = datetime.datetime.now()
        labels = []
        train_folder = self.root_path+'/Training/'
        test_folder = self.root_path+'/Testing/'

        quantity_tr = {} 
        quantity_te = {}
        for folder in os.listdir(train_folder):
            if folder != '.DS_Store':
                quantity_tr[folder] = len(os.listdir(train_folder+folder))
                labels.append(folder)
            
        for folder in os.listdir(test_folder):
            if folder != '.DS_Store':
                quantity_te[folder] = len(os.listdir(test_folder+folder))

        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[] 
        
        # Training set
        for label in labels:
            folder_dir = f'{train_folder}{label}'
            count = 0
            for file in os.listdir(folder_dir):
                img = cv2.imread(f'{folder_dir}/{file}')
                X_train.append(img)
                y_train.append(label)
                    

        # Testing set
        for label in labels:
            folder_dir = f'{test_folder}{label}'
            for file in os.listdir(folder_dir):
                img = cv2.imread(f'{folder_dir}/{file}')
                X_test.append(img)
                y_test.append(label)
                
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # Rescaling
        X_train=X_train/float(255.0)
        X_test=X_test/float(255.0)
        
        # Shuffling
        X_train_full, y_train_full = shuffle(X_train, y_train, random_state=42)
        self.X_test, y_test = shuffle(X_test, y_test, random_state=42)
        
        # One-hot encoding y labels
        y_train_full = tf.keras.utils.to_categorical([labels.index(i) for i in y_train_full])
        self.y_test = tf.keras.utils.to_categorical([labels.index(i) for i in y_test])
        
        # Creating validation datasets (500 length)
        self.X_train, self.y_train = X_train_full[:-500], y_train_full[:-500]
        self.X_val, self.y_val = X_train_full[-500:], y_train_full[-500:]

        t2 = datetime.datetime.now()

        # Shapes
        print(f'X_train shape: {self.X_train.shape}')
        print(f'y_train shape: {self.y_train.shape}')
        print(f'X_test shape: {self.X_test.shape}')
        print(f'y_test shape: {self.y_test.shape}')
        print(f'X_val shape: {self.X_val.shape}')
        print(f'y_val shape: {self.y_val.shape}')
        print(f'Data processed in: {t2-t1}')
        
        # return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val

   
    def build_model(self, num_conv_layers, num_dense_layers, save_results=True):
        t1 = datetime.datetime.now()
        print(f'Starting Conv: {num_conv_layers} - Dense: {num_dense_layers}')
        model = Sequential()
        
        # Convolution and pooling layers 
        if num_conv_layers == 1:
            model.add(layers.Conv2D(64, (3, 3), input_shape=(240, 240, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
        elif num_conv_layers == 2:
            model.add(layers.Conv2D(64, (3, 3), input_shape=(240, 240, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
            model.add(layers.Conv2D(64, (3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
        elif num_conv_layers == 3:
            model.add(layers.Conv2D(64, (3, 3), input_shape=(240, 240, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
            model.add(layers.Conv2D(64, (3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
            model.add(layers.Conv2D(128, (3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
        elif num_conv_layers == 4:
            model.add(layers.Conv2D(64, (3, 3), input_shape=(240, 240, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
            model.add(layers.Conv2D(64, (3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
            model.add(layers.Conv2D(128, (3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
            model.add(layers.Conv2D(128, (3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
        # Flattern the layers
        model.add(layers.Flatten())
            
        # Dense and Dropout layers
        if num_dense_layers == 1:
            model.add(layers.Dense(4, activation="softmax"))   
            
        elif num_dense_layers == 2:
            model.add(layers.Dense(256, activation="relu"))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(4, activation="softmax"))   
            
        elif num_dense_layers == 3:
            model.add(layers.Dense(256, activation="relu"))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(256, activation="relu"))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(4, activation="softmax"))  
            
        elif num_dense_layers == 4:
            model.add(layers.Dense(256, activation="relu"))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(256, activation="relu"))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(256, activation="relu"))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(4, activation="softmax")) 
            
        # Compiling, fitting and evaluating model
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])
        model_history = model.fit(self.X_train, self.y_train,
                    epochs=15,
                    verbose=1,
                    validation_data=(self.X_val, self.y_val))
        
        history_dict = model_history.history
        score = model.evaluate(self.X_test, self.y_test, verbose=0)
        t2 = datetime.datetime.now()
        
        print(f'Completed Conv{num_conv_layers} - Dense: {num_dense_layers}')
        print(f'TTR: {t2-t1}')
        
        result_dict = [{
            'time': str(t2-t1),
            'num_conv': num_conv_layers,
            'num_dense': num_dense_layers,
            'test_loss':score[0],
            'test_accuracy':score[1],
            'loss_values':history_dict['loss'],
            'val_loss':history_dict['val_loss']
        }]
            
        # Saving results
        if save_results == True:
            # If file not exists, create new file
            if os.path.exists("model_creation/model_structure_testing.json") == False:
                with open("model_creation/model_structure_testing.json", "w") as outfile:
                    json.dump(result_dict, outfile, indent=3)
                    print('new file made')
            else:
                # Open json file, append new contents
                with open("model_creation/model_structure_testing.json", "r") as infile:
                    data = json.load(infile)
                    data.extend(result_dict)
                    
                # Save extended results
                with open("model_creation/model_structure_testing.json", "w") as outfile:
                    json.dump(data, outfile, indent=3)
                    print('updated file')