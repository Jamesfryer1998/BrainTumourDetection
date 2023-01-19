import os
import datetime
from pre_processing import ImagePreProcessing


def preprocess_images():
    base_paths = ['Datasets', 'Datasets_cleaned/240_resolution']
    directories = [
        'brain_tumour_small/no',
        'brain_tumour_small/yes',
        'brain_tumour_large/Testing/glioma',
        'brain_tumour_large/Testing/meningioma',
        'brain_tumour_large/Testing/notumor',
        'brain_tumour_large/Testing/pituitary',
        'brain_tumour_large/Training/glioma',
        'brain_tumour_large/Training/meningioma',
        'brain_tumour_large/Training/notumor',
        'brain_tumour_large/Training/pituitary',
    ]

    count = 0
    new_dir_count = 0
    bad_img_count = 0
    t1 = datetime.datetime.now()

    for dir in directories:
        t3 = datetime.datetime.now()
        for file in os.listdir(f'{base_paths[0]}/{dir}'):
            count += 1
            try:
                ImagePreProcessing(f'{base_paths[0]}/{dir}/{file}', (240, 240), f'{base_paths[1]}/{dir}/{file}').pre_process_data()
            except OSError as error:
                bad_img_count += 1
            
        new_dir_count += len(os.listdir(f'{base_paths[1]}/{dir}'))

        t4 = datetime.datetime.now()
        print(f'{base_paths[0]}/{dir} - Processed - TTR: {t4-t3}')

    t2 = datetime.datetime.now()
    print('-------------------------------------------------------------------------------')
    print(f'{count} - Images Processed')
    print(f'{bad_img_count} - Unprocessed Images due to error.')
    print(f'All images processed in {t2-t1}')

    # Assert all images have been processed
    assert (count - bad_img_count) == new_dir_count

preprocess_images()