import datetime
import pandas as pd
import os 
import cv2
import time
import numpy as np


class ImagePreProcessing:
    def __init__(self, image):
        self.image = image
        self.cleaned_image = None


    def squishing(self):
        return


    def cropping(self):
        return


    def resizing(self, resolution):
        # this path will change
        path = '/Users/james/MScCode/Final Project/Datasets/test_dataset'

        if os.path.exists(f'{path}/features.npy') == True:
            os.remove(f'{path}/features.npy')
            time.sleep(0.1)

        training_data = []
        for img in os.listdir(path):
            pic = cv2.imread(os.path.join(path,img))
            pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
            pic = cv2.resize(pic,resolution)
            training_data.append([pic])

        np.save(os.path.join(path,'features'),np.array(training_data))

    def pro_process_data(self):
        self.squishing()
        self.cropping()
        self.resizing()

        return self.cleaned_image

ImagePreProcessing(1).resizing((300, 300))