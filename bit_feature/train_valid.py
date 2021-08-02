import numpy as np
import torch
# import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from nets import *
import time, os, copy, argparse
import multiprocessing
import json
# from torchsummary import summary
from matplotlib import pyplot as plt
# from model import *
# from focal_loss import FocalLoss
# from lr_scheduler import LR_Scheduler
import csv
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
import BiT_models

# Construct argument parser

# Set training mode
train_mode = 'finetune'
train_info = []
# Batch size
bs = 32
# Number of epochs
num_epochs = 100
# Note: Number of classes. Check before running
num_classes = 7
train_splits = 5
# Number of workers
num_cpu = multiprocessing.cpu_count()
# num_cpu = 0

train_directory = 'data_root/learning/training/'
model_dir = 'data_root/learning/models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if len(os.listdir(model_dir)) != 0:
    input('Model root not empty. Press Enter to continue...')

# Tensorboard summary
writer = SummaryWriter(log_dir='runs/Aug_1_train_all_new_2')

for val_folder_index in range(train_splits):  # Note: with validation: for val_folder_index in range(5):
    # whole_data_set = ['P17-2343;S6;UVM_R0_labeled_tiles', 'P17-4786;S5;UVM_R0_labeled_tiles',
    #                   'P17-7861;S4;UVM_R0_labeled_tiles', 'P17-8000;S2;UVM_R0_labeled_tiles',
    #                   'P18-6324;S2;UVM_R0_labeled_tiles', 'P18-8264;S2;UVM_R0_labeled_tiles',
    #                   'P16-7404;S6;UVM_R0_labeled_tiles', 'P16-8407;S8;UVM_R0_labeled_tiles',
    #                   'P16-8902;S6;UVM_R0_labeled_tiles', 'P16-8917;S6;UVM_R0_labeled_tiles',
    #                   'P17-2515;S5;UVM_R0_labeled_tiles', 'P17-2518;S6;UVM_R0_labeled_tiles',
    #                   'P17-2674;S6;UVM_R0_labeled_tiles', 'P17-4786;S5;UVM_R0_labeled_tiles',
    #                   'P17-4786;S6;UVM_R0_labeled_tiles']
    whole_data_set = [f'folder{i}' for i in range(1, 6)]
    val_idx_list = [[0], [1], [2], [3], [4]]
    val_WSI_list = [whole_data_set[i] for i in val_idx_list[val_folder_index]]  # Note: valid cmt
    for val in val_WSI_list:
        whole_data_set.remove(val)
    train_WSI_list = whole_data_set

    # Applying transforms to the data
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=256),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406],
            #                      [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
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

    # For code verification, loading the binary one
    # dataset['train'] = datasets.ImageFolder(root=train_directory + 'folder1', transform=image_transforms['train'])

    # Load validation data
    # loader_list = []
    # for slide in val_WSI_list:
    #     tmp_data = datasets.ImageFolder(root=train_directory + slide, transform=image_transforms['valid'])
    #     loader_list.append(tmp_data)
    # Note: valid cmt
    # dataset['valid'] = data.ConcatDataset(loader_list)
    dataset['valid'] = datasets.ImageFolder(root=train_directory + val_WSI_list[0], transform=image_transforms['valid'])

    # For code verification, loading the binary one
    # dataset['valid'] = datasets.ImageFolder(root='data_root/learning/testing', transform=image_transforms['valid'])

    # Size of train and validation data
    dataset_sizes = {
        'train': len(dataset['train']),
        'valid': len(dataset['valid'])  # Note: valid cmt
    }

    # Create iterators for data loading
    dataloaders = {
        'train': data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                                 num_workers=num_cpu, pin_memory=True, drop_last=True),
        # Note: valid cmt
        'valid': data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                                 num_workers=num_cpu, pin_memory=True, drop_last=True)
    }

    # Print the train and validation data sizes
    print("Training-set size:", dataset_sizes['train'],
          "\nValidation-set size:", dataset_sizes['valid'])  # Note: valid cmt

    # Saving the classifier labels
    # Note: does not work with ConcatDataset
    # train_labels = dataset['train'].class_to_idx
    valid_labels = dataset['valid'].class_to_idx
    # if train_labels != valid_labels:
    #     print('WARNING: training and validation class number mismatch!')
    with open('data_root/learning/models/train_labels.txt', 'w') as train_labels_file:
        train_labels_file.write(json.dumps(valid_labels))

    # Set default device as gpu, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if train_mode == 'finetune':
        # Load a pretrained model - BiT
        print("\nLoading BiT-M-R50x1 for finetuning ...\n")
        model = BiT_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
        model.load_from(np.load(f"{'BiT-M-R50x1'}.npz"))
        classifier = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    # Transfer the model to GPU
    model = torch.nn.DataParallel(model)
    classifier = torch.nn.DataParallel(classifier)
    model = model.to(device)
    classifier = classifier.to(device)

    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # Learning rate decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Model training routine
    print("\nTraining:-\n")

    since = time.time()

    best_acc = 0.0

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:  # Note: orig with valid: for phase in ['train', 'valid']:
            if phase == 'train':
                for param in classifier.parameters():
                    param.requires_grad = True
                classifier.train()
            else:
                for param in classifier.parameters():
                    param.requires_grad = False
                classifier.eval()

            running_loss = 0.0
            running_corrects = 0

            pred = []
            true = []

            # if need index: for i, (inputs, labels) in enumerate(dataloaders[phase], 0):
            for inputs, labels in dataloaders[phase]:
                classifier.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                _, feature = model(inputs)
                preds = classifier(feature)
                loss = F.cross_entropy(preds, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                preds_list = list(np.array(preds.argmax(dim=1).cpu().detach().numpy()))
                labels_list = list(np.array(labels.cpu().detach().numpy()))
                pred.append(preds_list)
                true.append(labels_list)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (np.array(pred) == np.array(true)).sum().item()

            if phase == 'train':
                scheduler.step()  # Note: was originally behind line 181. But should be here?
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            pred = sum(pred, [])
            true = sum(true, [])
            f1 = f1_score(true, pred, average='macro')
            balance_acc = balanced_accuracy_score(true, pred)

            print(f'{val_folder_index + 1}/4 {phase} Loss: {epoch_loss:.4f}, Acc: {balance_acc:.4f}')

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

            # deep copy the model
            # # Note: valid cmt
            if phase == 'valid' and balance_acc > best_acc:
                best_epoch = epoch
                best_split = val_folder_index
                best_acc = balance_acc
        PATH = f'{model_dir}/train_all_{val_folder_index}_epoch_' + str(epoch) + '.pth'
        torch.save(classifier.state_dict(), PATH)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))
    print(f'Best epoch in val is {best_epoch} in split {best_split}')
    # train_info.append([val_WSI_list, best_epoch, best_acc])
writer.close()
# with open('data_root/learning/training_output/train2_summary.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerows(train_info)

'''
Sample run: python train.py --mode=finetue
'''
