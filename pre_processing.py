import datetime
import pandas as pd

class ImagePreProcessing:
    def __init__(self, image):
        self.image = image
        self.cleaned_image = None

    def resizing(self):
        return

    def squishing(self):
        return

    def cropping(self):
        return

    def pro_process_data(self):
        self.resizing()
        self.squishing()
        self.cropping()

        return self.cleaned_image