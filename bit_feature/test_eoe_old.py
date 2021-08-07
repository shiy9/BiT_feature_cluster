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
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
# from focal_loss import FocalLoss
# from lr_scheduler import LR_Scheduler
import torch.nn as nn
import torch.nn.functional as F
import BiT_models

# Set the train and validation directory paths
test_directory = 'data_root/learning/testing/folder1'
# Set the model save path
classifierl1_model_path = 'data_root/learning/models_ctrst_flip/train_all_0_epoch_8_l1.pth'
classifierl2_model_path = 'data_root/learning/models_ctrst_flip/train_all_0_epoch_8_l2.pth'

# Batch size
bs = 32
# Number of classes
num_classes = 7
# Number of workers
# num_cpu = multiprocessing.cpu_count()
num_cpu = 0

# Applying transforms to the data
image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=256),
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

# Transfer the model to GPU
model = BiT_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
model.load_from(np.load(f"{'BiT-M-R50x1'}.npz"))
classifier_l1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
classifier_l2 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

model = torch.nn.DataParallel(model)
classifier_l1 = torch.nn.DataParallel(classifier_l1)
classifier_l2 = torch.nn.DataParallel(classifier_l2)

classifier_l1.load_state_dict(torch.load(classifierl1_model_path))
classifier_l2.load_state_dict(torch.load(classifierl2_model_path))

model = model.to(device)
classifier_l1 = classifier_l1.to(device)
classifier_l2 = classifier_l2.to(device)

since = time.time()
best_acc = 0.0

for param in model.parameters():
    param.requires_grad = False
model.eval()  # Set model to evaluate mode

for param in classifier_l1.parameters():
    param.requires_grad = False
classifier_l1.eval()

for param in classifier_l2.parameters():
    param.requires_grad = False
classifier_l2.eval()

running_corrects = 0

pred = []
true = []

# Note: indexing does not really work with shuffle=True. Order of samples/imgs in dataloader is not right
# Can definitely change to for inputs, labels in dataloaders['test']
for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    # forward
    _, feature = model(inputs)
    l1_out = classifier_l1(feature)
    l1_out = F.relu(l1_out)
    preds = classifier_l2(l1_out)
    # _, preds = torch.max(preds, 1)

    preds_list = list(np.array(preds.argmax(dim=1).cpu().detach().numpy()))
    labels_list = list(np.array(labels.cpu().detach().numpy()))
    pred.append(preds_list)
    true.append(labels_list)

    running_corrects += (np.array(preds_list) == np.array(labels_list)).sum().item()

# pred = sum(pred, [])
# true = sum(true, [])
epoch_acc = running_corrects / dataset_sizes['test']
cm = confusion_matrix(true, pred)

f1 = f1_score(true, pred, labels=[0, 1, 2, 3, 4, 5, 6], average='macro')
balance_acc = balanced_accuracy_score(true, pred)
print(f'Model testing balanced accuracy: {balance_acc:4f}')
print(f'model f1 score: {f1:4f}')
print('Confusion matrix: ')
print(cm)

time_elapsed = time.time() - since
print('Testing complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Test Acc: {:4f}'.format(epoch_acc))

'''
Sample run: python train.py --mode=finetue
'''
