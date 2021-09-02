import numpy as np
import torch
from torchvision import datasets, transforms, models
import torch.utils.data as data
import time
# from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import multiprocessing
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
import openslide
from PIL import Image

# Set the train and validation directory paths
test_directory = 'data_root/learning/unannot_test/P19-1500;S6;UVM_R0_tiles'
# Set the model save path
best_models = [14, 80, 14, 12, 41]
model_folder = 'models_finetune1l_inet_0.01'
save_name = ''

# Properties for the slide
patch_size = 256
reg_num = 0
downsample = 1
WSI_name = 'P19-1500;S6;UVM'
WSI_ext = '.scn'

# [bzh, dis, eos, fibrotic lp, normal lp, others, tissue]
# [red, green, blue, dark pink, light pink, gray, light purple]
colors_list = [(204, 0, 0), (0, 204, 0), (0, 0, 204), (102, 0, 102), (255, 51, 255), (105, 105, 105), (203, 195, 227)]

class1_pth = f'data_root/learning/{model_folder}/train_all_0_epoch_{best_models[0]}.pth'
class2_pth = f'data_root/learning/{model_folder}/train_all_1_epoch_{best_models[1]}.pth'
class3_pth = f'data_root/learning/{model_folder}/train_all_2_epoch_{best_models[2]}.pth'
class4_pth = f'data_root/learning/{model_folder}/train_all_3_epoch_{best_models[3]}.pth'
class5_pth = f'data_root/learning/{model_folder}/train_all_4_epoch_{best_models[4]}.pth'

# Batch size
bs = 1
# Number of classes
num_classes = 7
# Number of workers
# num_cpu = multiprocessing.cpu_count()
num_cpu = 0

# Applying transforms to the data
image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])
}

# Load data from folders
dataset = {
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

# Size of train and validation data
dataset_sizes = {
    'test': len(dataset['test'])
}

# Create iterators for data loading
dataloaders = {
    'test': data.DataLoader(dataset['test'], batch_size=bs, shuffle=False,
                            num_workers=num_cpu, pin_memory=True, drop_last=False)
}

# Class names or target labels
print(dataset['test'].class_to_idx)

# Print the train and validation data sizes
print("\nTesting-set size:", dataset_sizes['test'])

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_1 = models.resnet50(pretrained=True)
fc_features = model_1.fc.in_features
model_1.fc = nn.Linear(fc_features, num_classes)

model_2 = models.resnet50(pretrained=True)
fc_features = model_2.fc.in_features
model_2.fc = nn.Linear(fc_features, num_classes)

model_3 = models.resnet50(pretrained=True)
fc_features = model_3.fc.in_features
model_3.fc = nn.Linear(fc_features, num_classes)

model_4 = models.resnet50(pretrained=True)
fc_features = model_4.fc.in_features
model_4.fc = nn.Linear(fc_features, num_classes)

model_5 = models.resnet50(pretrained=True)
fc_features = model_5.fc.in_features
model_5.fc = nn.Linear(fc_features, num_classes)

# model_1 = torch.nn.DataParallel(model_1)
# model_2 = torch.nn.DataParallel(model_2)
# model_3 = torch.nn.DataParallel(model_3)
# model_4 = torch.nn.DataParallel(model_4)
# model_5 = torch.nn.DataParallel(model_5)

model_1.load_state_dict(torch.load(class1_pth))
model_2.load_state_dict(torch.load(class2_pth))
model_3.load_state_dict(torch.load(class3_pth))
model_4.load_state_dict(torch.load(class4_pth))
model_5.load_state_dict(torch.load(class5_pth))

model_1 = model_1.to(device)
model_2 = model_2.to(device)
model_3 = model_3.to(device)
model_4 = model_4.to(device)
model_5 = model_5.to(device)

since = time.time()
best_acc = 0.0

for param in model_1.parameters():
    param.requires_grad = False
model_1.eval()  # Set model to evaluate mode

for param in model_2.parameters():
    param.requires_grad = False
model_2.eval()

for param in model_3.parameters():
    param.requires_grad = False
model_3.eval()

for param in model_4.parameters():
    param.requires_grad = False
model_4.eval()

for param in model_5.parameters():
    param.requires_grad = False
model_5.eval()

tile_files = os.listdir(test_directory + '/eoe')

simg = openslide.open_slide('/home/yuxuanshi/VUSRP/WSI/' + WSI_name + WSI_ext)
max_w = int(simg.properties[f'openslide.region[{reg_num}].width'])  # matches x
max_h = int(simg.properties[f'openslide.region[{reg_num}].height'])  # matches y
img_x = int(simg.properties[f'openslide.region[{reg_num}].x'])
img_y = int(simg.properties[f'openslide.region[{reg_num}].y'])

img_x = int(int(np.floor(img_x / downsample / patch_size)) * downsample * patch_size)
img_y = int(int(np.floor(img_y / downsample / patch_size)) * downsample * patch_size)
max_w = int(int(np.ceil((img_x + max_w) / downsample / patch_size)) * downsample * patch_size) - img_x
max_h = int(int(np.ceil((img_y + max_h) / downsample / patch_size)) * downsample * patch_size) - img_y


print(f'{WSI_name}: Region {reg_num}    H: {max_h}  W: {max_w}')

# file_name_abbr = train_img_file.split('.')[0]
coor_file = os.path.join('data_root/tiles_coord/unannot_test', WSI_name + f'_R{reg_num}_tiles_coord.npy')

new_img_w = int(max_w/downsample//patch_size)
new_img_h = int(max_h/downsample//patch_size)
img_color = Image.new(mode = "RGB", size = (new_img_w, new_img_h))

img_color_arr = np.array(img_color).astype(np.uint8) + 255

coor_array = np.load(coor_file)

# Note: indexing does not really work with shuffle=True. Order of samples/imgs in dataloader is not right
# Can definitely change to for inputs, labels in dataloaders['test']
for i, (inputs, _) in enumerate(dataloaders['test'], 0):  # for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device, non_blocking=True)

    # majority vote
    cls1_prob = np.array(model_1(inputs).cpu().detach().numpy())
    cls2_prob = np.array(model_2(inputs).cpu().detach().numpy())
    cls3_prob = np.array(model_3(inputs).cpu().detach().numpy())
    cls4_prob = np.array(model_4(inputs).cpu().detach().numpy())
    cls5_prob = np.array(model_5(inputs).cpu().detach().numpy())

    norm_1 = preprocessing.normalize([cls1_prob[0]])
    norm_2 = preprocessing.normalize([cls2_prob[0]])
    norm_3 = preprocessing.normalize([cls3_prob[0]])
    norm_4 = preprocessing.normalize([cls4_prob[0]])
    norm_5 = preprocessing.normalize([cls5_prob[0]])
    norm_sum = norm_1 + norm_2 + norm_3 + norm_4 + norm_5
    pred = np.argmax(norm_sum)
    pred = pred.astype(int)

    coor_x = float(coor_array[i][0] - img_x) / downsample / patch_size
    coor_y = float(coor_array[i][1] - img_y) / downsample / patch_size

    color = colors_list[pred]
    img_color_arr[int(coor_y)][int(coor_x)] = color

Image.fromarray(img_color_arr).save(f'data_root/learning/unannot_test/{WSI_name}_R{reg_num}_cluster_mask.png')

