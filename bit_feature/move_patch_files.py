import os
import shutil
import sys
import random

# Create learning directory (destination)
learning_dir = 'data_root/learning'
if not os.path.exists(learning_dir):
    os.makedirs(learning_dir)

training_dir = f'{learning_dir}/training'
testing_dir = f'{learning_dir}/testing'
if not os.path.exists(training_dir):
    os.makedirs(training_dir)
if not os.path.exists(testing_dir):
    os.makedirs(testing_dir)

if len(os.listdir(training_dir)) + len(os.listdir(testing_dir)) != 0:
    sys.exit('Learning folder not empty. Aborting...')

# Slide name: region number
tiles_dir = 'data_root/tiles'
slide_dict = {'P16-8917;S6;UVM': 0, 'P17-2343;S6;UVM': 0, 'P17-4786;S5;UVM': 0, 'P17-7861;S4;UVM': 0,
              'P17-8000;S2;UVM': 0, 'P18-6324;S2;UVM': 0, 'P18-8264;S2;UVM': 0}
train_target_label_ct = {'bzh': 190, 'dis': 165, 'eos': 190, 'others': 0}
others_target_training_num = 500
others_target_test_num = 100
# others_target_num = {'P16-8917;S6;UVM': 73, 'P17-2343;S6;UVM': 124, 'P17-4786;S5;UVM': 45, 'P17-7861;S4;UVM': 63,
#                     'P17-8000;S2;UVM': 74, 'P18-6324;S2;UVM': 185, 'P18-8264;S2;UVM': 36}

temp_dir = f'{tiles_dir}/everything_delete'

for label in train_target_label_ct.keys():
    train_folder = f'{training_dir}/{label}'
    test_folder = f'{testing_dir}/{label}'
    temp_folder = f'{temp_dir}/{label}'
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

# Get the "others" label ready
for slide_folder, reg_num in slide_dict.items():
    if not os.path.exists(f'{tiles_dir}/{slide_folder}_R{reg_num}_labeled_tiles/others'):
        print(f'{slide_folder}_R{reg_num} does NOT have others folder!')
    else:
        parent_folder = f'{tiles_dir}/{slide_folder}_R{reg_num}_labeled_tiles'
        others_folder = parent_folder + '/others'
        destination_folder = f'{tiles_dir}/others_everything'
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        for file in os.listdir(others_folder):
            shutil.copy(f'{others_folder}/{file}', f'{destination_folder}/{file}')

# "others" training data
others_selected_filenames_train = random.sample(os.listdir(f'{tiles_dir}/others_everything'), others_target_training_num)
for file in others_selected_filenames_train:
    shutil.move(f'{tiles_dir}/others_everything/{file}', f'{training_dir}/others/{file}')

# "others" testing data
others_selected_filenames_test = random.sample(os.listdir(f'{tiles_dir}/others_everything'), others_target_test_num)
for file in others_selected_filenames_test:
    shutil.move(f'{tiles_dir}/others_everything/{file}', f'{testing_dir}/others/{file}')

# Cleanup
shutil.rmtree(f'{tiles_dir}/others_everything')

# Original tiles
for slide_folder, reg_num in slide_dict.items():
    for folder in os.listdir(f'{tiles_dir}/{slide_folder}_R{reg_num}_labeled_tiles'):
        cur_folder = f'{tiles_dir}/{slide_folder}_R{reg_num}_labeled_tiles/{folder}'
        if folder not in train_target_label_ct or folder == 'others':
            continue
        for file in os.listdir(cur_folder):
            shutil.copy(f'{cur_folder}/{file}', f'{training_dir}/{folder}/{file}')
            train_target_label_ct[folder] -= 1

# Super sampled tiles
for slide_folder, reg_num in slide_dict.items():
    for folder in os.listdir(f'{tiles_dir}/{slide_folder}_R{reg_num}_ss_tiles'):
        cur_folder = f'{tiles_dir}/{slide_folder}_R{reg_num}_ss_tiles/{folder}'
        if folder not in train_target_label_ct or folder == 'others':
            continue
        for file in os.listdir(cur_folder):
            shutil.copy(f'{cur_folder}/{file}', f'{temp_dir}/{folder}/{file}')

for folder in os.listdir(temp_dir):
    selected_files = random.sample(os.listdir(f'{temp_dir}/{folder}'), train_target_label_ct[folder])
    for file in selected_files:
        shutil.move(f'{temp_dir}/{folder}/{file}', f'{training_dir}/{folder}/{file}')

for folder in os.listdir(temp_dir):
    for file in os.listdir(f'{temp_dir}/{folder}'):
        shutil.copy(f'{temp_dir}/{folder}/{file}', f'{testing_dir}/{folder}/{file}')

# Clean up
shutil.rmtree(temp_dir)