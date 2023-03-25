# System
import os
import cv2
import json
import random
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers.legacy import Adam

# Keras
from keras.models import Sequential
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)

# Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

class TestSmallDataset:
    def __init__(self, root_path):
        self.root_path = root_path

    def process_data(self):
        t1 = datetime.datetime.now()

        all_files = []
        X = []
        y = []

        for folder in os.listdir(self.root_path):
            if folder != '.DS_Store':
                for file in os.listdir(f'{self.root_path}/{folder}'):
                    all_files.append(f'{self.root_path}/{folder}/{file}')
                    y.append(folder)

        # map values 0 to 'yes' and 1 to 'no' in y
        y = [0 if i == 'yes' else 1 for i in y]

        for x in all_files:
            X.append(cv2.imread(x))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        print(len(self.X_train))

        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        # Rescaling
        self.X_train=self.X_train/float(255.0)
        self.X_test=self.X_test/float(255.0)

        # Shuffling
        self.X_train_full, self.y_train_full = shuffle(self.X_train, self.y_train, random_state=42)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test, random_state=42)

        self.y_train_full = tf.keras.utils.to_categorical(self.y_train_full)
        self.y_test = tf.keras.utils.to_categorical(self.y_test)

        # Creating validation datasets (500 length)
        val_length = int(len(self.X_train) * 0.2)
        self.X_train, self.y_train = self.X_train_full[:-val_length], self.y_train_full[:-val_length]
        self.X_val, self.y_val = self.X_train_full[-val_length:], self.y_train_full[-val_length:]

        t2 = datetime.datetime.now()

        print(np.unique(self.y_test))

        # Shapes
        print(f'X_train shape: {self.X_train.shape}')
        print(f'y_train shape: {self.y_train.shape}')
        print(f'X_test shape: {self.X_test.shape}')
        print(f'y_test shape: {self.y_test.shape}')
        print(f'X_val shape: {self.X_val.shape}')
        print(f'y_val shape: {self.y_val.shape}')
        print(f'Data processed in: {t2-t1}')

    def build_model(self, conv_1_2_units, conv_3_4_units, dense_units, epoch, save_results=True, evaluation_vis=False, visualise_predictions=False):
        tf.keras.backend.clear_session()
        t1 = datetime.datetime.now()
        print(f'Starting run')
        model = Sequential()

        # Convolutional and max-pooling
        model.add(layers.Conv2D(conv_1_2_units, (3, 3), input_shape=(240, 240, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        model.add(BatchNormalization()) 
        model.add(layers.Conv2D(conv_1_2_units, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        model.add(BatchNormalization()) 
        model.add(layers.Conv2D(conv_3_4_units, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        model.add(BatchNormalization()) 
        model.add(layers.Conv2D(conv_3_4_units, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        # Flattern the layers
        model.add(layers.Flatten())

        # Dense layers and Dropout
        model.add(layers.Dense(dense_units, activation="relu"))
        model.add(layers.Dropout(0.7))
        model.add(layers.Dense(dense_units, activation="relu"))
        model.add(layers.Dropout(0.7))
        model.add(layers.Dense(2, activation="softmax"))  

        # Compiling, fitting and evaluating model
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])
        model_history = model.fit(self.X_train, self.y_train,
                    epochs=epoch,
                    verbose=1,
                    validation_data=(self.X_val, self.y_val))
        
        history_dict = model_history.history
        score = model.evaluate(self.X_test, self.y_test, verbose=0)
        t2 = datetime.datetime.now()

        # Classification
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(y_pred.argmax(axis=1), self.y_test.argmax(axis=1))

        accuracy = accuracy_score(y_pred.argmax(axis=1), self.y_test.argmax(axis=1))
        auc = roc_auc_score(self.y_test, y_pred)

        # Printing stats
        print(classification_report(y_pred.argmax(axis=1), self.y_test.argmax(axis=1)))
        print(f'Accuracy of model: {accuracy}')
        print(f'AUC of model: {auc}')

        if evaluation_vis == True:
            plt.figure(figsize = (10,7))
            ax = sns.heatmap(cm, annot=True)
            ax.set(xlabel="Class", ylabel="Class")
            
            print('Completed run')
            print(f'TTR: {t2-t1}')

            from sklearn.metrics import roc_curve, auc

            # Plotting ROC Curves per class
            # y_true contains the true class labels, y_score contains the predicted probabilities or scores
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_pred[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot the ROC curves for each class
            plt.figure()
            for i in range(2):
                plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic for multi-class')
            plt.legend(loc="lower right")
            plt.show()

        if visualise_predictions:
            # Create a 4x4 subplot
            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            axes = axes.ravel()
            index_to_labels = {0: 'yes', 1: 'no'}
            
            # Get a random selection of "yes" and "no" images
            yes_indices = np.where(np.argmax(self.y_test, axis=1) == 0)[0]
            no_indices = np.where(np.argmax(self.y_test, axis=1) == 1)[0]
            yes_sample = random.sample(list(yes_indices), 8)
            no_sample = random.sample(list(no_indices), 8)
            sample_indices = yes_sample + no_sample
            random.shuffle(sample_indices)

            # Loop through the sample images and their predictions
            for i, index in enumerate(sample_indices):
                # Get the actual and predicted values
                actual = np.argmax(self.y_test[index])
                pred = np.argmax(y_pred[index])
                color = 'green' if actual == pred else 'red'

                # Display the image and its actual/predicted class
                axes[i].imshow(self.X_test[index])
                axes[i].set_title(f"Actual: {index_to_labels[actual]}\nPredicted: {index_to_labels[pred]}", color=color, fontsize=10)
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()
        
        result_dict = [{
            'time': str(t2-t1),
            'num_conv_layers': 4,
            'num_dense_layers': 3,
            'conv_1_2_unit': conv_1_2_units,
            'conv_3_4_unit': conv_3_4_units,
            'dense_unit': dense_units,
            'epoch': epoch,
            'test_loss':score[0],
            'test_accuracy':score[1],
            'loss_values':history_dict['loss'],
            'val_loss':history_dict['val_loss']
        }]

        print(result_dict)
        
        # Saving results
        if save_results == True:
            # If file not exists, create new file
            if os.path.exists("hyperparameter_testing/final_model_testing.json") == False:
                with open("hyperparameter_testing/final_model_testing.json", "w") as outfile:
                    json.dump(result_dict, outfile, indent=3)
                    print('new file made')
            else:
                # Open json file, append new contents
                with open("hyperparameter_testing/final_model_testing.json", "r") as infile:
                    data = json.load(infile)
                    data.extend(result_dict)
                    
                # Save extended results
                with open("hyperparameter_testing/final_model_testing.json", "w") as outfile:
                    json.dump(data, outfile, indent=3)
                    print('updated file')