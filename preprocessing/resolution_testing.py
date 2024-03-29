import os
import datetime
from pre_processing import ImagePreProcessing

t1 = datetime.datetime.now()

def preprocess_resolutions_testing():
    resolution_dict = {
        '32': (32, 32),
        '64': (64, 64),
        '128': (128, 128),
        '240': (240, 240),
        '320': (320,320),
    }

    base_directory = 'Datasets_cleaned/brain_tumour_small'
    type = ['yes', 'no']
    bad_image_count = 0

    for type in type:
        for res_dir, res in resolution_dict.items():
            for file in os.listdir(f'{base_directory}/{type}'):
                try:
                    ImagePreProcessing(f'{base_directory}/{type}/{file}', res, f'Datasets_cleaned/Resolutions_testing/{res_dir}/{type}/{file}').pre_process_data()
                except OSError as error:
                    bad_image_count += 1

            print(f'{res_dir} - {type} tumour - Processed.')
    
    print(f'{bad_image_count} bad images found.')

    t2 = datetime.datetime.now()
    print(t2-t1)