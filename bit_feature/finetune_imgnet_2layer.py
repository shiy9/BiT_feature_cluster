import numpy as np
import torch
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os, copy, argparse
import multiprocessing
import json
from matplotlib import pyplot as plt
import csv
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

train_info = []
bs = 256
num_epochs = 100
# Note: Number of classes. Check before running
num_classes = 7
train_splits = 5
lr = 1e-3
stepSize = 15
save_name = 'finetune2l_imgnet_2aug_w1'
# Number of workers
num_cpu = multiprocessing.cpu_count()
# num_cpu = 0

train_directory = 'data_root/learning/training/'
model_dir = f'data_root/learning/models_{save_name}'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if len(os.listdir(model_dir)) != 0:
    input('Model root not empty. Press Enter to continue...')

# Tensorboard summary
writer = SummaryWriter(log_dir=f'runs/Aug_10_{save_name}')

since = time.time()

for val_folder_index in range(train_splits):
    whole_data_set = [f'folder{i}' for i in range(1, 6)]
    val_idx_list = [[0], [1], [2], [3], [4]]
    val_WSI_list = [whole_data_set[i] for i in val_idx_list[val_folder_index]]
    for val in val_WSI_list:
        whole_data_set.remove(val)
    train_WSI_list = whole_data_set

    # Applying transforms to the data
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.ColorJitter(contrast=(1.0, 1.5)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406],
            #                      [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor()
        ])
    }

    # Load data from folders
    dataset = {}
    # Load training data
    loader_list = []
    for slide in train_WSI_list:
        tmp_data = datasets.ImageFolder(root=train_directory + slide, transform=image_transforms['train'])
        loader_list.append(tmp_data)
    dataset['train'] = data.ConcatDataset(loader_list)

    dataset['valid'] = datasets.ImageFolder(root=train_directory + val_WSI_list[0], transform=image_transforms['valid'])

    # Size of train and validation data
    dataset_sizes = {
        'train': len(dataset['train']),
        'valid': len(dataset['valid'])
    }

    # Create iterators for data loading
    dataloaders = {
        'train': data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                                 num_workers=num_cpu, pin_memory=True, drop_last=False),
        'valid': data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                                 num_workers=num_cpu, pin_memory=True, drop_last=False)
    }

    # Print the train and validation data sizes
    print("Training-set size:", dataset_sizes['train'],
          "\nValidation-set size:", dataset_sizes['valid'])

    # Saving the classifier labels
    # Note: does not work with ConcatDataset
    # train_labels = dataset['train'].class_to_idx
    valid_labels = dataset['valid'].class_to_idx
    # if train_labels != valid_labels:
    #     print('WARNING: training and validation class number mismatch!')
    with open(f'{model_dir}/train_labels.txt', 'w') as train_labels_file:
        train_labels_file.write(json.dumps(valid_labels))

    # Set default device as gpu, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 1024)

    classifier_l2 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    for param in classifier_l2.parameters():
        param.requires_grad = True

    model = torch.nn.DataParallel(model)
    classifier_l2 = torch.nn.DataParallel(classifier_l2)

    # Transfer the model to GPU
    model = model.to(device)
    classifier_l2 = classifier_l2.to(device)

    optimizer = optim.SGD(model.module.fc.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer_l2 = optim.SGD(classifier_l2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=stepSize, gamma=0.1)
    scheduler_l2 = lr_scheduler.StepLR(optimizer_l2, step_size=stepSize, gamma=0.1)

    # Model training routine
    print("\nTraining:-\n")

    # def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Load the previous epoch's model in before training
        # For debugging only. Comment out later!
        for phase in ['train', 'valid']:  # Note: orig with valid: for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                classifier_l2.train()
            else:
                model.eval()
                classifier_l2.eval()

            running_loss = 0.0
            running_corrects = 0

            pred = []
            true = []

            # if need index: for i, (inputs, labels) in enumerate(dataloaders[phase], 0):
            for inputs, labels in dataloaders[phase]:
                model.zero_grad()
                classifier_l2.zero_grad()
                optimizer.zero_grad()
                optimizer_l2.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)

                l1_out = model(inputs)
                l1_out = torch.nn.functional.relu(l1_out)
                preds = classifier_l2(l1_out)

                weights = torch.tensor([0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4])
                weights = weights.to(device)

                loss = F.cross_entropy(preds, labels, weight=weights)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer_l2.step()

                preds_list = list(np.array(preds.argmax(dim=1).cpu().detach().numpy()))
                labels_list = list(np.array(labels.cpu().detach().numpy()))
                pred.append(preds_list)
                true.append(labels_list)

                # statistics
                running_loss += loss.item() * inputs.size(0)

                # First line wrong?
                # running_corrects += (np.array(pred) == np.array(true)).sum().item()
                running_corrects += (np.array(preds_list) == np.array(labels_list)).sum().item()

            if phase == 'train':
                scheduler.step()
                scheduler_l2.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            pred = sum(pred, [])
            true = sum(true, [])
            f1 = f1_score(true, pred, average='macro')
            balance_acc = balanced_accuracy_score(true, pred)

            print(f'{val_folder_index + 1}/{train_splits} {phase} Loss: {epoch_loss:.4f}, Acc: {balance_acc:.4f}')

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar(f'Train{val_folder_index}/Loss', epoch_loss, epoch)
                writer.add_scalar(f'Train{val_folder_index}/Accuracy', epoch_acc, epoch)
                writer.add_scalar(f'Train{val_folder_index}/F1_score', f1, epoch)
                writer.add_scalar(f'Train{val_folder_index}/Balance_accuracy', balance_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar(f'Valid{val_folder_index}/Loss', epoch_loss, epoch)
                writer.add_scalar(f'Valid{val_folder_index}/Accuracy', epoch_acc, epoch)
                writer.add_scalar(f'Valid{val_folder_index}/F1_score', f1, epoch)
                writer.add_scalar(f'Valid{val_folder_index}/Balance_accuracy', balance_acc, epoch)
                writer.flush()

            if phase == 'valid' and balance_acc > best_acc:
                best_epoch = epoch
                best_split = val_folder_index
                best_acc = balance_acc
        PATH = f'{model_dir}/train_all_{val_folder_index}_epoch_' + str(epoch) + '_l1.pth'
        torch.save(model.state_dict(), PATH)
        PATH = f'{model_dir}/train_all_{val_folder_index}_epoch_' + str(epoch) + '_l2.pth'
        torch.save(classifier_l2.state_dict(), PATH)

    print('Best val Acc: {:4f}'.format(best_acc))
    print(f'Best epoch in val is {best_epoch} in split {best_split}')
    train_info.append([val_WSI_list, best_epoch, best_acc])
writer.close()
with open(f'{model_dir}/training_summary.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(train_info)

time_elapsed = time.time() - since
print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

'''
Sample run: python train.py --mode=finetue
'''
