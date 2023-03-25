# System and images
import os
import cv2
import json
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
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)


class MultiClassModelCreation:
    def __init__(self, root_path):
        self.root_path = root_path

    def process_data(self):
        self.index_to_labels = {0: 'pituitary', 1: 'notumor', 2: 'glioma', 3: 'meningioma'}
        t1 = datetime.datetime.now()
        labels = []
        self.test_paths = []
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
                self.test_paths.append(f'{folder_dir}/{file}')
                
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
        
   
    def build_model(self, conv_1_2_units, conv_3_4_units, dense_units, epoch,
                    save_results=True, evaluation_vis=False, visualise_predictions=False,
                    grouped_classification=False):
        
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
        model.add(layers.Dense(4, activation="softmax"))  

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

        if grouped_classification == True:
            grouped_actual = [0 if test in [0,2,3] else 1 for test in self.y_test.argmax(axis=1)]
            grouped_preds = [0 if test in [0,2,3] else 1 for test in y_pred.argmax(axis=1)]

            cm = confusion_matrix(grouped_preds, grouped_actual)
            plt.figure(figsize = (10,7))
            ax = sns.heatmap(cm, annot=True)
            ax.set(xlabel="Class", ylabel="Class")
            plt.show()

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
            for i in range(4):
                fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_pred[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot the ROC curves for each class
            plt.figure()
            for i in range(4):
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
            index_to_labels = {0: 'pituitary', 1: 'notumor', 2: 'glioma', 3: 'meningioma'}

            # Loop through the images and their predictions
            for i in range(len(self.X_test)):
                # Get the actual and predicted values
                actual = np.argmax(self.y_test[i])
                pred = np.argmax(y_pred[i])
                color = 'green' if actual == pred else 'red'

                # Display the image and its actual/predicted class
                axes[i].imshow(self.X_test[i])
                axes[i].set_title(f"Actual: {index_to_labels[actual]}\nPredicted: {index_to_labels[pred]}", color=color, fontsize=10)
                axes[i].axis('off')

                # Stop the loop if all subplots are displayed
                if i == 15:
                    break
                
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