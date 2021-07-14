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
# from torchsummary import summary
# from matplotlib import pyplot as plt
# from model import *
from sklearn.metrics import confusion_matrix, f1_score
# from focal_loss import FocalLoss
# from lr_scheduler import LR_Scheduler
import BiT_models

# Set the train and validation directory paths
# train_directory = '/share/contrastive_learning/resnet50_v2/imds_small/imdb_small/train'
# valid_directory = '/share/contrastive_learning/resnet50_v2/imds_small/imdb_small/val'

# train_directory = '/share/contrastive_learning/resnet50_v2/data_0122/train_patch'
# valid_directory = '/share/contrastive_learning/resnet50_v2/data_0122/val_patch'
test_directory = 'data_root/learning/testing/P18-8264;S2;UVM_R0_labeled_tiles/'
# test_directory = '/share/contrastive_learning/data/sup_data/data_0124_10000/test_patch'
# Set the model save path
classifier_model_path = 'data_root/learning/best_model/train1_epoch_12.pth'

# Batch size
bs = 1
# Number of classes
num_classes = 3
# Number of workers
num_cpu = multiprocessing.cpu_count()
# num_cpu = 0

# Applying transforms to the data
image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
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
    'test': data.DataLoader(dataset['test'], batch_size=bs, shuffle=True,
                             num_workers=num_cpu, pin_memory=True, drop_last=True)
}

# Class names or target labels
print(dataset['test'].class_to_idx)

# Print the train and validation data sizes
print("\nTesting-set size:", dataset_sizes['test'])

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transfer the model to GPU
model = BiT_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
model.load_from(np.load(f"{'BiT-M-R50x1'}.npz"))
classifier = torch.load(classifier_model_path).to(device)

model = model.to(device)
classifier = classifier.to(device)

since = time.time()
best_acc = 0.0

model.eval()  # Set model to evaluate mode
classifier.eval()
running_corrects = 0

pred = []
true = []
for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    # forward
    _, feature = model(inputs)
    preds = classifier(feature)
    _, preds = torch.max(preds, 1)

    running_corrects += torch.sum(preds == labels.data)

    preds_list = list(np.array(preds.cpu()))
    labels_list = list(np.array(labels.cpu()))
    pred.append(preds_list)
    true.append(labels_list)

pred = sum(pred, [])
true = sum(true, [])
epoch_acc = running_corrects.double() / dataset_sizes['test']
cm = confusion_matrix(true, pred)
f1 = f1_score(true, pred, labels=[0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15 ], average='macro')
print('model f1 score:  ', f1)
print('Confusion matrix: ')
print(cm)
np.savetxt("data_root/learning/testing_output/cm_train1_12.csv", cm, delimiter=",")
time_elapsed = time.time() - since
print('Testing complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Test Acc: {:4f}'.format(epoch_acc))

'''
Sample run: python train.py --mode=finetue
'''
