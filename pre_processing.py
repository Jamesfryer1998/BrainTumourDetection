import datetime
import pandas as pd
import os 
import cv2
import time
import numpy as np
from PIL import Image, ImageEnhance

# TODO:
# - Crop images in test set
# - Rezise images in test set
# - Play around with distortion of images

class ImagePreProcessing:
    def __init__(self, image_path):
        self.image_path = image_path
        self.cleaned_image = None


    def distortion(self, on_off='on'):
        # Rabbit hole

        return 

    def cropping(self):
        img_org = Image.open(self.image_path)
        img = img_org.convert("L")
        pixels = img.load()

        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if pixels[i,j] != (25):
                    pixels[i,j] = (0)
            
        bbox = img.getbbox()
        cropped_im = img_org.crop(bbox)

        # Save the cropped image
        cropped_im.save(self.image_path)
        
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
        self.distortion()
        self.cropping()
        self.resizing()

        return self.cleaned_image

# Resizing
# ImagePreProcessing('Datasets/test_dataset/Te-gl_0010.jpg').resizing((300, 300))

# Cropping
# ImagePreProcessing('Datasets/test_dataset/Te-gl_0020 copy.jpg').cropping()

directory = 'Datasets/test_dataset_processed'

t1 = datetime.datetime.now()
for file in os.listdir(directory):
    ImagePreProcessing(f'{directory}/{file}').cropping()

t2 = datetime.datetime.now()
print(t2-t1)