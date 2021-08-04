import math
import os
import shutil
import random

sorting_dir = 'data_root/learning/training'

for i in range(2, 6):
    if not os.path.exists(f'{sorting_dir}/folder{i}'):
        os.makedirs(f'{sorting_dir}/folder{i}')
    else:
        input('Destination sorting folders already exists, press Enter to continue: ')

for folder in os.listdir(f'{sorting_dir}/folder1'):
    files = os.listdir(f'{sorting_dir}/folder1/{folder}')
    random.shuffle(files)
    target_num = math.ceil(len(files) * 0.2)
    for i in range(2, 6):
        moving = files[:target_num]
        for file in moving:
            if not os.path.exists(f'{sorting_dir}/folder{i}/{folder}'):
                os.makedirs(f'{sorting_dir}/folder{i}/{folder}')
            shutil.move(f'{sorting_dir}/folder1/{folder}/{file}', f'{sorting_dir}/folder{i}/{folder}/{file}')
        del files[:target_num]
