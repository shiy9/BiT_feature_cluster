import numpy as np
import torch
# import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
# from torch.utils.tensorboard import SummaryWriter
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from nets import *
import time, os, copy, argparse
import multiprocessing
import BiT_models
import BiT_models_1024
# from matplotlib import pyplot as plt
# from model import *
from sklearn import preprocessing
# from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
# from torchvision.models import resnet50, resnet18

patch_size = 256
WSI_name = 'P16-7404;S6;UVM'
reg_num = 4
feature_1024 = True

save_ext = '_1024' if feature_1024 else ''

feature_dict = {}
coord_dict = {}
# valid_directory = '/share/contrastive_learning/data/data_w_nearby_patches/original_4w'
save_folder = f'data_root/feature/'
valid_directory = f'data_root/tiles/{WSI_name}_R{reg_num}_tiles/'
# train_directory = '/share/contrastive_learning/resnet50_v2/data_0122/train_patch'
# valid_directory = '/share/contrastive_learning/resnet50_v2/data_0122/val_patch'
# valid_directory = '/share/contrastive_learning/data/crop_after_process_doctor/merged_data_test_no_minor/'
# test_directory = '/share/contrastive_learning/data/sup_data/data_0124_10000/test_patch'
# Set the model save path
# model_path = '/share/contrastive_learning/resnet50_v2/pytorch-image-classification-master/checkpoint/exp_0208/train_0208_v0_84.pth'

# Batch size
bs = 1
# Number of workers
num_cpu = multiprocessing.cpu_count()
# num_cpu = 0

# Applying transforms to the data
image_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        transforms.Resize(size=patch_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=patch_size),
        # transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

dataset_val = datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])

# Size of train and validation data
dataset_sizes = {
    'valid': len(dataset_val)
}

# Create iterators for data loading
dataloaders = {
    'valid': data.DataLoader(dataset_val, batch_size=bs, shuffle=False,
                             num_workers=num_cpu, pin_memory=True, drop_last=False)
}

# Print the train and validation data sizes
print("\nValidation-set size:", dataset_sizes['valid'])

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MODEL = '/share/contrastive_learning/SimSiam_PatrickHua/simsiam-v4-3path/checkpoint/simsiam-TCGA-0218-nearby_0223173249.pth'
# # Load the model for testing
# backbone = 'resnet50'
# backbone = eval(f"{backbone}()")
# backbone.output_dim = backbone.fc.in_features
# backbone.fc = torch.nn.Identity()
# model = backbone
# save_dict = torch.load(MODEL, map_location='cpu')
# model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
#                             strict=True)

##  BiT model load pretrain param
if feature_1024:
    model = BiT_models_1024.KNOWN_MODELS['BiT-M-R50x1'](head_size=9, zero_head=True)
else:
    model = BiT_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=9, zero_head=True)
toy_model = np.load(f"{'BiT-M-R50x1'}.npz")
model.load_from(np.load(f"{'BiT-M-R50x1'}.npz"))

since = time.time()
best_acc = 0.0

model.to(device).eval()  # Set model to evaluate mode
i = 0

feature_arr = []

for inputs, labels in dataloaders['valid']:  # TODO: original for inputs, labels, path in dataloaders['valid']:
    inputs = inputs.to(device, non_blocking=True)
    i += 1
    if feature_1024:
        feature = model(inputs)
    else:
        _, feature = model(inputs)
    feature_arr.append(feature.detach().cpu().numpy())
    # WSI_name = path[0].split('/')[-1].split('_')[0]
    # x_coord = path[0].split('/')[-1].split('_')[1]
    # y_coord = path[0].split('/')[-1].split('_')[2].split('.')[0]

    # if WSI_name in feature_dict.keys():
    #     feature_dict[WSI_name] = np.vstack((feature_dict[WSI_name], feature_arr))
    #     # coord_dict[WSI_name] = np.vstack((coord_dict[WSI_name], [x_coord, y_coord]))
    # else:
    #     feature_dict[WSI_name] = feature_arr
    #     # coord_dict[WSI_name] = [x_coord, y_coord]
    # if i % 100 == 0:
    #     print('Image: ' + str(i).zfill(5))

# for WSI in feature_dict.keys():
#     fea = feature_dict[WSI]
#     coor = coord_dict[WSI]
#     np.save(save_folder + 'fea1/' + WSI + '_fea.npy', fea)
#     np.save(save_folder + 'coor1/' + WSI + '_coor.npy', coor)

feature_arr = np.squeeze(feature_arr, axis=1)
np.save(f'{save_folder}{WSI_name}_R{reg_num}_feature{save_ext}.npy', feature_arr)
# np.savetxt("cm_0221_triple_200.csv", cm, delimiter=",")
time_elapsed = time.time() - since
print('Feature extraction complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''
Sample run: python train.py --mode=finetue
'''
