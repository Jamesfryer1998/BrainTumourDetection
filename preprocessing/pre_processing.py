import datetime
import pandas as pd
import os 
import numpy as np
from PIL import Image
import re

# TODO:
# - Create model and test different resolutions
# - Play around with distortion of images

class ImagePreProcessing:
    def __init__(self, image_path, resolution, save_path=None):
        self.image_path = image_path
        self.save_path = save_path
        self.resolution = resolution
        self.cleaned_image = None

    def distortion(self, on_off='on'):
        # Rabbit hole

        return 

    def cropping(self):
        img_org = Image.open(self.image_path)

        # Converting image to greyscale, makes pixel map
        pixels = np.array(img_org.convert('L'))

        # Changing dark pixels to black
        pixels[pixels < 50] = 0

        # Cropping image with greyscale bbox
        cropped_image = img_org.crop(Image.fromarray(pixels).getbbox())

        # Assigning cropped image
        self.cleaned_image = cropped_image
        
    def resizing(self):
        resized_img = self.cleaned_image.resize(self.resolution)

        if self.save_path == None:
            resized_img.save(self.image_path)
        else:
            resized_img.save(self.save_path)
        
    def pre_process_data(self):
        # self.distortion()
        self.cropping()
        self.resizing()

        return self.cleaned_image

directories = ['Datasets_cleaned/test_dataset_processed/no', 'Datasets_cleaned/test_dataset_processed/yes']

t1 = datetime.datetime.now()
for directory in directories:
    for file in os.listdir(directory):
        ImagePreProcessing(f'{directory}/{file}', (300,300)).pre_process_data()

t2 = datetime.datetime.now()
print(t2-t1)

# list_directories = [
#     'Datasets_cleaned/brain_tumour_small/no',
#     'Datasets_cleaned/brain_tumour_small/yes',
#     'Datasets_cleaned/brain_tumour_large/Testing/glioma',
#     'Datasets_cleaned/brain_tumour_large/Testing/meningioma',
#     'Datasets_cleaned/brain_tumour_large/Testing/notumor',
#     'Datasets_cleaned/brain_tumour_large/Testing/pituitary',
#     'Datasets_cleaned/brain_tumour_large/Training/glioma',
#     'Datasets_cleaned/brain_tumour_large/Training/meningioma',
#     'Datasets_cleaned/brain_tumour_large/Training/notumor',
#     'Datasets_cleaned/brain_tumour_large/Training/pituitary',
# ]

# count = 0

# for directory in list_directories:
#     count += len(os.listdir(directory))

# print(count)
