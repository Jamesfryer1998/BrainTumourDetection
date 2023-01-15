import os
import datetime
from pre_processing import ImagePreProcessing

t1 = datetime.datetime.now()

resolution_dict = {
    '32': (32, 32),
    '64': (64, 64),
    '128': (128, 128),
    '240': (240, 240),
    '320': (320,320),
    '640': (640, 640)
}

def process_resolutions(resolutions):

    base_directory = 'Datasets_cleaned/brain_tumour_small'
    type = ['yes', 'no']

    for type in type:
        for res_dir, res in resolutions.items():
            print(res)
            print(res_dir)

            for file in os.listdir(f'{base_directory}/{type}'):
                ImagePreProcessing(f'{base_directory}/{type}/{file}', res, f'Datasets_cleaned/Resolutions/{res_dir}/{type}/{file}').pre_process_data()

    t2 = datetime.datetime.now()
    print(t2-t1)

# process_resolutions(resolution_dict)

# TODO:
# - Ready to run, will process no/yes images in brain_tumour_small dataset
# - Wills save files with different resolutions in relavent resolutions directories
# - Once completed, run a method for each of the resolutions and comapre results