import os

import numpy as np
import openslide
import cv2
import os
import shutil
import random

# Crop out interested region. For 7/13/21 Presentation
# data = np.load('data_root/tiles_coord/P16-8917;S6;UVM_R0_tiles_coord.npy')
# coord = data[32]
# coord[0] = coord[0] - 512
# coord[1] = coord[1] - 512
# wsi = openslide.open_slide('/home/yuxuanshi/VUSRP/WSI/Processed/P16-8917;S6;UVM.scn')
# size = 6 * 512
# region = wsi.read_region(coord, 0, (size, size))
# region = np.array(region.convert('RGB'))
# cv2.imwrite('ReportImgs/temp_scrshot.png', region)

# Copy one single file
file_dir = 'data_root/learning/training/folder1/background'
# for i in range(200):
#     shutil.copy(f'{file_dir}/P17-4786;S5;UVM_010905.png', f'{file_dir}/P17-4786;S5;UVM_010905_{i}.png')

# Copy a bunch of files
selected_files = random.sample(os.listdir(file_dir), 2)
for file in selected_files:
    shutil.copy(f'{file_dir}/{file}', f'{file_dir}/copy_{file}')



print('dummy print')