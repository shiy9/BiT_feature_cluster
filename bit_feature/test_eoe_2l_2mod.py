import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import time
import imgaug.augmenters as iaa
# from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
import torch.nn.functional as F
import torch.nn as nn
from sklearn import preprocessing
import multiprocessing
import BiT_models

# Set the train and validation directory paths
test_directory = 'data_root/learning/testing_flip_sharp/folder1'

best_models = [13, 7, 5, 52, 50]

class1_l1_pth = f'data_root/learning/models/train_all_0_epoch_{best_models[0]}_l1.pth'
class1_l2_pth = f'data_root/learning/models/train_all_0_epoch_{best_models[0]}_l2.pth'

class2_l1_pth = f'data_root/learning/models/train_all_1_epoch_{best_models[1]}_l1.pth'
class2_l2_pth = f'data_root/learning/models/train_all_1_epoch_{best_models[1]}_l2.pth'

class3_l1_pth = f'data_root/learning/models/train_all_2_epoch_{best_models[2]}_l1.pth'
class3_l2_pth = f'data_root/learning/models/train_all_2_epoch_{best_models[2]}_l2.pth'

class4_l1_pth = f'data_root/learning/models/train_all_3_epoch_{best_models[3]}_l1.pth'
class4_l2_pth = f'data_root/learning/models/train_all_3_epoch_{best_models[3]}_l2.pth'

class5_l1_pth = f'data_root/learning/models/train_all_4_epoch_{best_models[4]}_l1.pth'
class5_l2_pth = f'data_root/learning/models/train_all_4_epoch_{best_models[4]}_l2.pth'


save_name = '40x_2l_32_0.0001_weighted_1'

# Batch size
bs = 128
# Number of classes
num_classes = 7
# Number of workers
num_cpu = multiprocessing.cpu_count()
# num_cpu = 0

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

# Transfer the model to GPU
model = BiT_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
model.load_from(np.load(f"{'BiT-M-R50x1'}.npz"))

classifier1_l1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
classifier1_l2 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

classifier2_l1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
classifier2_l2 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

classifier3_l1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
classifier3_l2 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

classifier4_l1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
classifier4_l2 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

classifier5_l1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
classifier5_l2 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

model = torch.nn.DataParallel(model)

classifier1_l1 = torch.nn.DataParallel(classifier1_l1)
classifier1_l2 = torch.nn.DataParallel(classifier1_l2)

classifier2_l1 = torch.nn.DataParallel(classifier2_l1)
classifier2_l2 = torch.nn.DataParallel(classifier2_l2)

classifier3_l1 = torch.nn.DataParallel(classifier3_l1)
classifier3_l2 = torch.nn.DataParallel(classifier3_l2)

classifier4_l1 = torch.nn.DataParallel(classifier4_l1)
classifier4_l2 = torch.nn.DataParallel(classifier4_l2)

classifier5_l1 = torch.nn.DataParallel(classifier5_l1)
classifier5_l2 = torch.nn.DataParallel(classifier5_l2)

# classifier1.load_state_dict(torch.load(class1_pth))
# classifier2.load_state_dict(torch.load(class2_pth))
# classifier3.load_state_dict(torch.load(class3_pth))
# classifier4.load_state_dict(torch.load(class4_pth))
# classifier5.load_state_dict(torch.load(class5_pth))

classifier1_l1.load_state_dict(torch.load(class1_l1_pth))
classifier1_l2.load_state_dict(torch.load(class1_l2_pth))

classifier2_l1.load_state_dict(torch.load(class2_l1_pth))
classifier2_l2.load_state_dict(torch.load(class2_l2_pth))

classifier3_l1.load_state_dict(torch.load(class3_l1_pth))
classifier3_l2.load_state_dict(torch.load(class3_l2_pth))

classifier4_l1.load_state_dict(torch.load(class4_l1_pth))
classifier4_l2.load_state_dict(torch.load(class4_l2_pth))

classifier5_l1.load_state_dict(torch.load(class5_l1_pth))
classifier5_l2.load_state_dict(torch.load(class5_l2_pth))

model = model.to(device)

classifier1_l1 = classifier1_l1.to(device)
classifier1_l2 = classifier1_l2.to(device)

classifier2_l1 = classifier2_l1.to(device)
classifier2_l2 = classifier2_l2.to(device)

classifier3_l1 = classifier3_l1.to(device)
classifier3_l2 = classifier3_l2.to(device)

classifier4_l1 = classifier4_l1.to(device)
classifier4_l2 = classifier4_l2.to(device)

classifier5_l1 = classifier5_l1.to(device)
classifier5_l2 = classifier5_l2.to(device)

since = time.time()
best_acc = 0.0

for param in model.parameters():
    param.requires_grad = False
model.eval()  # Set model to evaluate mode

for param in classifier1_l1.parameters():
    param.requires_grad = False
classifier1_l1.eval()
for param in classifier1_l2.parameters():
    param.requires_grad = False
classifier1_l2.eval()

for param in classifier2_l1.parameters():
    param.requires_grad = False
classifier2_l1.eval()
for param in classifier2_l2.parameters():
    param.requires_grad = False
classifier2_l2.eval()

for param in classifier3_l1.parameters():
    param.requires_grad = False
classifier3_l1.eval()
for param in classifier3_l2.parameters():
    param.requires_grad = False
classifier3_l2.eval()

for param in classifier4_l1.parameters():
    param.requires_grad = False
classifier4_l1.eval()
for param in classifier4_l2.parameters():
    param.requires_grad = False
classifier4_l2.eval()

for param in classifier5_l1.parameters():
    param.requires_grad = False
classifier5_l1.eval()
for param in classifier5_l2.parameters():
    param.requires_grad = False
classifier5_l2.eval()

running_corrects = 0

pred = []
true = []

# Note: indexing does not really work with shuffle=True. Order of samples/imgs in dataloader is not right
# Can definitely change to for inputs, labels in dataloaders['test']
# for i, (inputs, labels) in enumerate(dataloaders['test'], 0):  # for inputs, labels in dataloaders['test']:
for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    # # Note: trouble shooting, delete later
    # spl_filename = dataloaders['test'].dataset.samples[i][0]
    # spl_filename = spl_filename.rsplit('/', 1)[-1]

    # forward
    _, feature = model(inputs)

    # Original
    # preds = classifier(feature)
    # _, preds = torch.max(preds, 1)

    label_arr = np.array(labels.cpu().detach().numpy())
    dis_bs = len(label_arr)

    # majority vote
    preds = np.zeros(dis_bs)

    cls1_l1 = classifier1_l1(feature)
    cls1_l1 = nn.functional.relu(cls1_l1)
    cls1_prob = np.array(classifier1_l2(cls1_l1).cpu().detach().numpy())

    cls2_l1 = classifier2_l1(feature)
    cls2_l1 = nn.functional.relu(cls2_l1)
    cls2_prob = np.array(classifier2_l2(cls2_l1).cpu().detach().numpy())

    cls3_l1 = classifier3_l1(feature)
    cls3_l1 = nn.functional.relu(cls3_l1)
    cls3_prob = np.array(classifier3_l2(cls3_l1).cpu().detach().numpy())

    cls4_l1 = classifier4_l1(feature)
    cls4_l1 = nn.functional.relu(cls4_l1)
    cls4_prob = np.array(classifier4_l2(cls4_l1).cpu().detach().numpy())

    cls5_l1 = classifier5_l1(feature)
    cls5_l1 = nn.functional.relu(cls5_l1)
    cls5_prob = np.array(classifier5_l2(cls5_l1).cpu().detach().numpy())


    for idx in range(dis_bs):
        norm_1 = preprocessing.normalize([cls1_prob[idx]])
        norm_2 = preprocessing.normalize([cls2_prob[idx]])
        norm_3 = preprocessing.normalize([cls3_prob[idx]])
        norm_4 = preprocessing.normalize([cls4_prob[idx]])
        norm_5 = preprocessing.normalize([cls5_prob[idx]])
        norm_sum = norm_1 + norm_2 + norm_3 + norm_4 + norm_5
        temp = np.argmax(norm_sum)
        preds[idx] = temp

    # running_corrects += torch.sum(preds == labels.data)
    # class_idx = preds.data[0]

    # preds_list = list(np.array(preds.cpu()))
    # labels_list = list(np.array(labels.cpu()))

    # preds_list = list(preds.astype(int))
    # labels_list = list(np.array(labels.cpu().detach().numpy()))

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
print(f'\nmodel f1 score: {f1:4f}')
print('Confusion matrix: ')
print(cm)
print(f'Testing accuracy: {epoch_acc:4f}')
balance_acc = balanced_accuracy_score(true, pred)
print(f'Balance accuracy: {balance_acc:4f}')
np.savetxt(f"data_root/learning/testing_output/train_{save_name}_cm.csv", cm, delimiter=",")

time_elapsed = time.time() - since
print('Testing complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''
Sample run: python train.py --mode=finetue
'''
