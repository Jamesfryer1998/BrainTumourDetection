print('Importing TensorFlow packages...')
from pre_processing import ImagePreProcessing
import os
import json
import time
import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Changes Tensoflow initalisation print outs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
print('Packages Successfully Imported.')


class ResolutionTesting:
    def __init__(self, res_dir, res, run):
        self.res_dir = res_dir
        self.res = res
        self.run = run
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def proccess_data(self):
        no_path = f'/Users/james/MScCode/Final Project/Datasets_cleaned/Resolutions/{self.res_dir}/no'
        yes_path = f'/Users/james/MScCode/Final Project/Datasets_cleaned/Resolutions/{self.res_dir}/yes'

        if len(os.listdir(no_path)) == 0 and len(os.listdir(yes_path)) == 0:
            raise Exception('No files present. Please pre-process data.')

        data = []
        result = []
        encoder = OneHotEncoder()
        encoder.fit([[0], [1]]) 

        # No tumour
        for file in os.listdir(no_path):

            img = Image.open(f'{no_path}/{file}')
            img = np.array(img)
            if img.shape == self.res:
                data.append(np.array(img))
                result.append(encoder.transform([[0]]).toarray())

        # Yes tumour
        for file in os.listdir(yes_path):
            img = Image.open(f'{yes_path}/{file}')
            img = np.array(img)
            if img.shape == self.res:
                data.append(np.array(img))
                result.append(encoder.transform([[1]]).toarray())

        data = np.array(data)
        results = np.array(result)
        results = results.reshape(data.shape[0], 2)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, results, test_size=0.3, shuffle=True, random_state=42)

        # return X_train, X_test, y_train, y_test

    def compile_model(self):
        model = Sequential()

        # res need to be in form of (x, y, z) e.g (128, 128, 3)
        model.add(Conv2D(32, kernel_size=(2, 2), input_shape=self.res, padding = 'Same'))
        model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))

        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
        model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])
        print(f'{self.res_dir} model compiling...')
        history = model.fit(self.X_train, self.y_train, epochs=50, batch_size=40, verbose=0,validation_data=(self.X_test, self.y_test))
        score = model.evaluate(self.X_test, self.y_test, verbose=0)
        pred = model.predict(self.X_test)
        y_pred = np.argmax(pred, axis=1)
        y_test_int = np.argmax(self.y_test, axis=1)
        confusion = confusion_matrix(y_test_int, y_pred)
        tf.keras.backend.clear_session()
        print(f'{self.res_dir} model compiled sucessfully.')

        # Save results
        save_dict = [{
            'resolution': self.res_dir,
            'run': self.run,
            'test_loss': score[0],
            'test_accuracy': score[1],
            'training_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'true_neg': int(confusion[0][0]),
            'false_neg': int(confusion[1][0]),
            'true_pos': int(confusion[1][1]),
            'false_pos': int(confusion[0][1])
        }]

        # If file not exists, create new file
        if os.path.exists("res_testing.json") == False:
            with open("res_testing.json", "w") as outfile:
                json.dump(save_dict, outfile, indent=3)
                print('     New file created.')
        else:
            # Open json file, append new contents
            with open("res_testing.json", "r") as infile:
                data = json.load(infile)
                data.extend(save_dict)
                
            # Save extended results
            with open("res_testing.json", "w") as outfile:
                json.dump(data, outfile, indent=3)
                print(f'    Resolution data updated.')

        print(f'Accuracy: {score[1]}')

    def run_all(self):
        self.proccess_data()
        self.compile_model()

def test_resolutions(run):
    print('Resolution testing starting...')
    resolution_dict = {
        '32': (32, 32, 3),
        '64': (64, 64, 3),
        '128': (128, 128, 3),
        '240': (240, 240, 3),
        '320': (320, 320, 3),
        # '400': (400, 400, 3)
    }

    for res_dir, res in resolution_dict.items():
        t1 = datetime.datetime.now()
        ResolutionTesting(res_dir, res, run).run_all()
        t2 = datetime.datetime.now()
        print(f'{res_dir} tested in {t2-t1}')
        time.sleep(3)