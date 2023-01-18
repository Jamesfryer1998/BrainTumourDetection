import os

base_dir = 'Datasets_cleaned/Resolutions'
directories = ['32', '64', '128', '320', '400']
types = ['yes', 'no']

for type in types:
    for directory in directories:
        for file in os.listdir(f'{base_dir}/{directory}/{type}'):
            os.remove(f'{base_dir}/{directory}/{type}/{file}')