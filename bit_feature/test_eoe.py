import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import time
import imgaug.augmenters as iaa
# from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
import torch.nn as nn
from sklearn import preprocessing
import BiT_models

# Set the train and validation directory paths
test_directory = 'data_root/learning/testing/folder1'
# Set the model save path
class1_pth = 'data_root/learning/models/train_all_0_epoch_5.pth'
class2_pth = 'data_root/learning/models/train_all_1_epoch_6.pth'
class3_pth = 'data_root/learning/models/train_all_2_epoch_14.pth'
class4_pth = 'data_root/learning/models/train_all_3_epoch_7.pth'
class5_pth = 'data_root/learning/models/train_all_4_epoch_9.pth'


# Batch size
bs = 64
# Number of classes
num_classes = 7
# Number of workers
# num_cpu = multiprocessing.cpu_count()
num_cpu = 0

# Applying transforms to the data
image_transforms = {
    'test': transforms.Compose([
        # transforms.Resize(size=256),

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
classifier1 = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
classifier2 = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
classifier3 = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
classifier4 = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
classifier5 = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

model = torch.nn.DataParallel(model)
classifier1 = torch.nn.DataParallel(classifier1)
classifier2 = torch.nn.DataParallel(classifier2)
classifier3 = torch.nn.DataParallel(classifier3)
classifier4 = torch.nn.DataParallel(classifier4)
classifier5 = torch.nn.DataParallel(classifier5)

classifier1.load_state_dict(torch.load(class1_pth))
classifier2.load_state_dict(torch.load(class2_pth))
classifier3.load_state_dict(torch.load(class3_pth))
classifier4.load_state_dict(torch.load(class4_pth))
classifier5.load_state_dict(torch.load(class5_pth))

model = model.to(device)

classifier1 = classifier1.to(device)
classifier2 = classifier2.to(device)
classifier3 = classifier3.to(device)
classifier4 = classifier4.to(device)
classifier5 = classifier5.to(device)

since = time.time()
best_acc = 0.0

for param in model.parameters():
    param.requires_grad = False
model.eval()  # Set model to evaluate mode

for param in classifier1.parameters():
    param.requires_grad = False
classifier1.eval()

for param in classifier2.parameters():
    param.requires_grad = False
classifier2.eval()

for param in classifier3.parameters():
    param.requires_grad = False
classifier3.eval()

for param in classifier4.parameters():
    param.requires_grad = False
classifier4.eval()

for param in classifier5.parameters():
    param.requires_grad = False
classifier5.eval()

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
    cls1_prob = np.array(classifier1(feature).cpu().detach().numpy())
    cls2_prob = np.array(classifier2(feature).cpu().detach().numpy())
    cls3_prob = np.array(classifier3(feature).cpu().detach().numpy())
    cls4_prob = np.array(classifier4(feature).cpu().detach().numpy())
    cls5_prob = np.array(classifier5(feature).cpu().detach().numpy())

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
print('model f1 score:  ', f1)
print('Confusion matrix: ')
print(cm)
print(f'Testing accuracy: {epoch_acc:4f}')
balance_acc = balanced_accuracy_score(true, pred)
print(f'Balance accuracy: {balance_acc:4f}')
np.savetxt("data_root/learning/testing_output/cm_train_0801.csv", cm, delimiter=",")

time_elapsed = time.time() - since
print('Testing complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

'''
Sample run: python train.py --mode=finetue
'''
