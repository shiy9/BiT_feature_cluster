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


# Set the train and validation directory paths
test_directory = 'data_root/learning/testing/folder1'
# Set the model save path
best_models = [15, 64, 11, 37, 48]
model_folder = 'models_finetune2l_inet_0.01_4aug'
save_name = 'finetune2l_inet_0.01_4aug'
plot_title = 'ResNet50 2-layer Finetune, 4 Augmentations'

class1_pth = f'data_root/learning/{model_folder}/train_all_0_epoch_{best_models[0]}.pth'
class2_pth = f'data_root/learning/{model_folder}/train_all_1_epoch_{best_models[1]}.pth'
class3_pth = f'data_root/learning/{model_folder}/train_all_2_epoch_{best_models[2]}.pth'
class4_pth = f'data_root/learning/{model_folder}/train_all_3_epoch_{best_models[3]}.pth'
class5_pth = f'data_root/learning/{model_folder}/train_all_4_epoch_{best_models[4]}.pth'

# Batch size
bs = 128
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
    'test': data.DataLoader(dataset['test'], batch_size=bs, shuffle=True,
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

running_corrects = 0

pred = []
true = []

# Note: indexing does not really work with shuffle=True. Order of samples/imgs in dataloader is not right
# Can definitely change to for inputs, labels in dataloaders['test']
# for i, (inputs, labels) in enumerate(dataloaders['test'], 0):  # for inputs, labels in dataloaders['test']:
for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    label_arr = np.array(labels.cpu().detach().numpy())
    dis_bs = len(label_arr)

    # majority vote
    preds = np.zeros(dis_bs)
    cls1_prob = np.array(model_1(inputs).cpu().detach().numpy())
    cls2_prob = np.array(model_2(inputs).cpu().detach().numpy())
    cls3_prob = np.array(model_3(inputs).cpu().detach().numpy())
    cls4_prob = np.array(model_4(inputs).cpu().detach().numpy())
    cls5_prob = np.array(model_5(inputs).cpu().detach().numpy())

    for idx in range(dis_bs):
        norm_1 = preprocessing.normalize([cls1_prob[idx]])
        norm_2 = preprocessing.normalize([cls2_prob[idx]])
        norm_3 = preprocessing.normalize([cls3_prob[idx]])
        norm_4 = preprocessing.normalize([cls4_prob[idx]])
        norm_5 = preprocessing.normalize([cls5_prob[idx]])
        norm_sum = norm_1 + norm_2 + norm_3 + norm_4 + norm_5
        temp = np.argmax(norm_sum)
        preds[idx] = temp

    preds_arr = preds.astype(int)

    running_corrects += np.sum(preds_arr == label_arr)
    print('.', end='')
    pred.extend(list(preds_arr))
    true.extend(list(label_arr))

# pred = sum(pred, [])
# true = sum(true, [])
epoch_acc = running_corrects / dataset_sizes['test']
cm = confusion_matrix(true, pred)
f1 = f1_score(true, pred, labels=[0, 1, 2, 3, 4, 5, 6], average='macro')
print('Confusion matrix: ')
print(cm)
print(f'\nTesting accuracy: {epoch_acc:4f}')
balance_acc = balanced_accuracy_score(true, pred)
print(f'Balance accuracy: {balance_acc:4f}')
print(f'model f1 score: {f1:4f}')
np.savetxt(f"data_root/learning/testing_output/{save_name}_cm.csv", cm, fmt='%i', delimiter=",")

time_elapsed = time.time() - since
print('Testing complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

index = ['bzh', 'dis', 'eos', 'fibrotic lp', 'normal lp', 'others', 'tissue']

cm_df = pd.DataFrame(cm, index=index, columns=index)
plt.figure(figsize=(9, 9))
ax = sn.heatmap(cm_df, annot=True, cmap='Blues', fmt='d', cbar=False, square=True, annot_kws={'fontsize':12})
ax.xaxis.tick_top()
plt.yticks(rotation=0)
plt.tick_params(axis='both', which='major', labelsize=12,
                labelbottom = False, bottom=False, top = False, left=False, labeltop=True)
plt.title(plot_title, fontdict={'fontsize': 15}, y=1.08)
plt.savefig(f'data_root/learning/testing_output/{save_name}_cm.png')
plt.show()

