import os

base_dir = 'Datasets_cleaned/Resolutions'
directories = ['32', '64', '128', '240', '320', '640']

for directory in directories:
    for file in os.listdir(f'{base_dir}/{directory}'):
        os.remove(f'{base_dir}/{directory}/{file}')