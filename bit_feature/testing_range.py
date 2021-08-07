import os
import numpy as np
import openslide
import cv2
import os
import shutil
import random
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
import imageio

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

# For testing
# whole_data_set = ['P17-2343;S6;UVM_R0_labeled_tiles', 'P17-4786;S5;UVM_R0_labeled_tiles',
#                   'P17-7861;S4;UVM_R0_labeled_tiles', 'P17-8000;S2;UVM_R0_labeled_tiles',
#                   'P18-6324;S2;UVM_R0_labeled_tiles', 'P18-8264;S2;UVM_R0_labeled_tiles',
#                   'P16-7404;S6;UVM_R0_labeled_tiles', 'P16-8407;S8;UVM_R0_labeled_tiles',
#                   'P16-8902;S6;UVM_R0_labeled_tiles', 'P16-8917;S6;UVM_R0_labeled_tiles',
#                   'P17-2343;S6;UVM_R0_labeled_tiles', 'P17-2515;S5;UVM_R0_labeled_tiles',
#                   'P17-2518;S6;UVM_R0_labeled_tiles', 'P17-2674;S6;UVM_R0_labeled_tiles',
#                   'P17-4786;S5;UVM_R0_labeled_tiles', 'P17-4786;S6;UVM_R0_labeled_tiles']
# val_idx_list = [[2, 7, 13, 6, 0], [6, 9], [14, 15, 3, 5, 6, 0], [10, 8, 6, 0]]
# val_WSI_list = [whole_data_set[i] for i in val_idx_list[1]]  # Note: valid cmt
# for val in val_WSI_list:
#     whole_data_set.remove(val)
# train_WSI_list = whole_data_set

# print('Haha here we go')
# input("Press Enter to continue")


# Randomly select a few patches and delete others
# target_num = 14
# others_dir = 'data_root/learning/training/P18-8264;S2;UVM_R0_labeled_tiles/others'
# name = 'P18-8264;S2;UVM_R0_labeled_tiles'
# destination_dir = ''
# files = os.listdir(others_dir)
# the_ones = random.sample(files, target_num)
# for file in the_ones:
#     files.remove(file)
# for file in files:
#     os.remove(f'{others_dir}/{file}')
# print(f'{name} Done!')

# Compare two masks
# img1 = Image.open('data_root/learning/testing/Test_17_4786_mask/P17-4786;S5;UVM_R0_trained_mask.png')
# img2 = Image.open('data_root/result_img/P17-4786;S5;UVM_R0_patch_group_mask.png')
#
# ct = 0
# all = 0
# pixels_1 = np.array(img1.getdata())
# pixels_2 = np.array(img2.getdata())
#
# size = pixels_1.shape
#
# for i in range(size[0]):
#     temp = pixels_1[i]
#     if pixels_1[i].any() != pixels_2[i].any():
#         ct += 1
# print(f'Difference {ct}')

# Data augmentation test
# img = imageio.imread('/home/yuxuanshi/VUSRP/big_transfer/bit_feature/data_root/learning/training/folder1/eos/P16-7404;S6;UVM_024858.png')
# transform = iaa.Sequential([
#     iaa.contrast.LinearContrast((1.0, 1.5))
#     ])
# img_aug = transform(images=img)
# ia.imshow(img)
# ia.imshow(img_aug)

# Data augmentation test 2 with opencv
img = cv2.imread('data_root/learning/testing/folder1/eos/P17-2674;S6;UVM_10.png')
cv2.imshow('original', img)
cv2.waitKey(0)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
ctrst = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('contrast', ctrst)
cv2.waitKey(0)
row,col,ch = ctrst.shape
mean = 0
var = 0.1
sigma = var**0.5
gauss = np.random.normal(mean,sigma,(row,col,ch))
gauss = gauss.reshape(row,col,ch)
noisy = ctrst + gauss
noisy = noisy.astype('uint8')
cv2.imshow('result', noisy)
cv2.imwrite('/home/yuxuanshi/VUSRP/big_transfer/bit_feature/data_root/P16-7404;S6;UVM_024858_c.png', noisy)
cv2.waitKey(0)

print('dummy print')