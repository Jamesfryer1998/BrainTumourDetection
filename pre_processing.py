import datetime
import pandas as pd
import os 
import cv2
import time
import numpy as np
from PIL import Image

# TODO:
# - Create model and test different resolutions
# - Play around with distortion of images

class ImagePreProcessing:
    def __init__(self, image_path, resolution):
        self.image_path = image_path
        self.resolution = resolution
        self.cleaned_image = None

    def distortion(self, on_off='on'):
        # Rabbit hole

        return 

    def cropping(self):
        img_org = Image.open(self.image_path)

        # Converting image to greyscale
        pixels = np.array(img_org.convert('L'))

        # Changing dark pixels to black
        pixels[pixels < 50] = 0

        # Cropping image with greyscale bbox
        cropped_image = img_org.crop(Image.fromarray(pixels).getbbox())

        # Assigning cropped image
        self.cleaned_image = cropped_image
        
    def resizing(self):
        resized_img = self.cleaned_image.resize(self.resolution)
        resized_img.save(self.image_path)
        
    def pro_process_data(self):
        # self.distortion()
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
    ImagePreProcessing(f'{directory}/{file}', (400,400)).pro_process_data()

t2 = datetime.datetime.now()
print(t2-t1)