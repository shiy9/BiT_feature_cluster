import os
import numpy as np
import openslide
import cv2
import os
import shutil
import random

import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import BiT_models

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
# file_dir = 'data_root/learning/training/folder1/background'
# for i in range(200):
#     shutil.copy(f'{file_dir}/P17-4786;S5;UVM_010905.png', f'{file_dir}/P17-4786;S5;UVM_010905_{i}.png')

# Copy a bunch of files
# selected_files = random.sample(os.listdir(file_dir), 2)
# for file in selected_files:
#     shutil.copy(f'{file_dir}/{file}', f'{file_dir}/copy_{file}')


# Compare two feature dictionaries
# train_fea_dict = np.load('data_root/learning/models/file_feature/train3_epoch_1_fea.npy', allow_pickle=True).item()
# train_2_fea_dict = np.load('data_root/learning/models/file_feature/train3_epoch_2_fea.npy', allow_pickle=True).item()
#
# for key, value in train_fea_dict.items():
#     ids = map(id, train_2_fea_dict.values())
#     if id(value) not in ids:
#         print('Feature different!')
#
# test_fea_dict = np.load('data_root/learning/models/file_feature/test3_epoch_24_fea_cp.npy', allow_pickle=True).item()
# test_2_fea_dict = np.load('data_root/learning/models/file_feature/test3_epoch_24_fea.npy', allow_pickle=True).item()
#
# for key, value in train_fea_dict.items():
#     test_fea = test_fea_dict[key]
#     dist = np.linalg.norm(value - test_fea)
#     print(f'{key} feature has difference {dist}')


# Model feature for one same photo
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_1 = BiT_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=1, zero_head=True)
# model_1.load_from(np.load(f"{'BiT-M-R50x1'}.npz"))
# model_1 = model_1.to(device)
# model_2 = BiT_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=1, zero_head=True)
# model_2.load_from(np.load(f"{'BiT-M-R50x1'}.npz"))
# model_2 = model_1.to(device)
#
# model_1.eval()
# model_2.eval()
#
# image_transform = transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
# ])
#
# dataset_1 = datasets.ImageFolder(root='data_root/just_a_fucking_test', transform=image_transform)
# dataloader_1 = data.DataLoader(dataset_1, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
#
# dataset_2 = datasets.ImageFolder(root='data_root/just_a_fucking_test', transform=image_transform)
# dataloader_2 = data.DataLoader(dataset_2, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
# feature_1 = {}
# feature_2 = {}
#
# for i, (inputs, _) in enumerate(dataloader_1, 0):
#     inputs = inputs.to(device, non_blocking=True)
#     spl_filename_1 = dataloader_1.dataset.samples[i][0]
#     spl_filename_1 = spl_filename_1.rsplit('/', 1)[-1]
#     # forward
#     _, feature = model_1(inputs)
#     feature_1[spl_filename_1] = feature.cpu().detach().numpy()
#
# for i, (inputs, _) in enumerate(dataloader_2, 0):
#     inputs = inputs.to(device, non_blocking=True)
#     spl_filename_2 = dataloader_2.dataset.samples[i][0]
#     spl_filename_2 = spl_filename_2.rsplit('/', 1)[-1]
#     # forward
#     _, feature = model_2(inputs)
#     feature_2[spl_filename_2] = feature.cpu().detach().numpy()


# Extract a few patches
WSI_name = 'P17-2515;S5;UVM'
reg_num = 0
coord_file = np.load(f'data_root/tiles_coord/{WSI_name}_R{reg_num}_tiles_coord.npy')
coord_1 = coord_file[79]
coord_2 = coord_file[89]
coord_1[0] = coord_1[0] + 512
coord_2[0] = coord_2[0] + 512
coord_file = list(coord_file)
coord_file.append(np.array([coord_1[0], coord_1[1]]))
coord_file.append(np.array([coord_2[0], coord_2[1]]))
wsi = openslide.open_slide(f'/home/yuxuanshi/VUSRP/WSI/{WSI_name}.scn')
size = 512

fnames_now = sorted(os.listdir(f'data_root/tiles/{WSI_name}_R{reg_num}_tiles'))
save_num = int(fnames_now[-1].rsplit('_', 1)[-1][:-4])
file_1 = wsi.read_region(coord_1, 0, (size, size))
file_1 = file_1.convert('RGB')
save_num += 1
file_1.save(f'data_root/tiles/{WSI_name}_R{reg_num}_tiles/{WSI_name}_{save_num}.png')  # Note: change

file_2 = wsi.read_region(coord_2, 0, (size, size))
file_2 = file_2.convert('RGB')
save_num += 1
file_2.save(f'data_root/tiles/{WSI_name}_R{reg_num}_tiles/{WSI_name}_{save_num}.png')  # Note: change
np.save(f'data_root/tiles_coord/{WSI_name}_R{reg_num}_tiles_coord.npy', coord_file)



print('dummy print')