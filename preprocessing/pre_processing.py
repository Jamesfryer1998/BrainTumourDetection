import datetime
import pandas as pd
import os 
import numpy as np
from PIL import Image
import re

# TODO:


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