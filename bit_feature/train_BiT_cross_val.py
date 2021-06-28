import numpy as np
import torch
# import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
# from nets import *
import time, os, copy, argparse
import multiprocessing
# from torchsummary import summary
# from matplotlib import pyplot as plt
# from model import *
# from focal_loss import FocalLoss
# from lr_scheduler import LR_Scheduler
import csv
# from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.metrics import f1_score, balanced_accuracy_score
# import microsoftvision
import BiT_models

# Construct argument parser

# Set training mode
train_mode = 'finetune'
train_info = []
# Batch size
bs = 1
# Number of epochs
num_epochs = 100
# Number of classes
num_classes = 9
# Number of workers
num_cpu = multiprocessing.cpu_count()
# num_cpu = 0


patch_size = 256
WSI_name = 'P16-7404;S6;UVM'
reg_num = 4


for val_folder_index in range(1):  # original: for val_folder_index in range(5):
    # whole_train_list = ['D8E6', '117E', '676F', 'E2D7', 'BE52']
    # val_WSI_list = whole_train_list[val_folder_index]
    # train_WSI_list = whole_train_list
    # train_WSI_list.pop(val_folder_index)
    # val_WSI_list = 'test_R4'

    train_directory = f'data_root/tiles/{WSI_name}_R{reg_num}_tiles/'
    '''valid_directory = '/share/contrastive_learning/data/crop_after_process_doctor/merged_data_no_minor/'''''

    # train_directory = '/share/contrastive_learning/data/crop_after_process_doctor/partial_training_data/1percent/'
    # valid_directory = '/share/contrastive_learning/data/crop_after_process_doctor/partial_training_data/1percent/'
    # Set the model save path
    # best_PATH = "models/train_beta.pth"

    # Applying transforms to the data
    image_transforms = {
        # TODO: orig cmt
        # 'train': transforms.Compose([
        #     transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        #     transforms.Resize(size=128),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406],
        #                          [0.229, 0.224, 0.225])
        # ]),

        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=patch_size, scale=(1.0, 1.0)),
            transforms.Resize(size=patch_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        # TODO: orig cmt
        # 'valid': transforms.Compose([
        #     # transforms.Resize(size=128),
        #     transforms.Resize(size=128),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406],
        #                          [0.229, 0.224, 0.225])
        # ])
    }

    # Load data from folders

    dataset = {}
    # TODO: maybe get rid of the transforms?
    dataset_train0 = datasets.ImageFolder(root=train_directory, transform=image_transforms['train'])
    # dataset_train0 = datasets.ImageFolder(root=train_directory)
    dataset['train'] = dataset_train0

    # TODO: orig cmt
    # dataset_train0 = datasets.ImageFolder(root=train_directory + train_WSI_list[0], transform=image_transforms['train'])
    # dataset_train1 = datasets.ImageFolder(root=train_directory + train_WSI_list[1], transform=image_transforms['train'])
    # dataset_train2 = datasets.ImageFolder(root=train_directory + train_WSI_list[2], transform=image_transforms['train'])
    # dataset_train3 = datasets.ImageFolder(root=train_directory + train_WSI_list[3], transform=image_transforms['train'])
    # dataset['valid'] = datasets.ImageFolder(root=valid_directory + val_WSI_list, transform=image_transforms['valid'])
    # dataset['train'] = data.ConcatDataset([dataset_train0, dataset_train1, dataset_train2, dataset_train3])

    # Size of train and validation data
    dataset_sizes = {
        'train': len(dataset['train']),
        # TODO: orig cmt: 'valid': len(dataset['valid'])
    }

    # Create iterators for data loading
    dataloaders = {
        'train': data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                                 num_workers=num_cpu, pin_memory=True, drop_last=True),
        # TODO: orig cmt
        # 'valid': data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
        #                          num_workers=num_cpu, pin_memory=True, drop_last=True)
    }

    # Print the train and validation data sizes
    # TODO: orig cmt
    # print("Training-set size:", dataset_sizes['train'],
    #       "\nValidation-set size:", dataset_sizes['valid'])
    print("Feature extraction size:", dataset_sizes['train'])

    # Set default device as gpu, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if train_mode == 'finetune':
        # Load a pretrained model - Resnet18
        print("\nLoading resnet50 for finetuning ...\n")
        # model_ft = models.resnet18(pretrained=True)
        # model = microsoftvision.models.resnet50(pretrained=True)
        # classifier = nn.Linear(in_features=2048, out_features=9, bias=True).to(device)

        model = BiT_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
        model.load_from(np.load(f"{'BiT-M-R50x1'}.npz"))

    elif train_mode == 'scratch':
        model_ft = models.resnet50(pretrained=False)
        fc_features = model_ft.fc.in_features
        # correct class number
        model_ft.fc = nn.Linear(fc_features, num_classes)
        # Set number of epochs to a higher value

    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    # Transfer the model to GPU
    model = model.to(device)

    # Print model summary
    print('Model Summary:-\n')

    # Loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(class_num=num_classes)
    # Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=0.1,
    #                       momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05, momentum=0.9)
    # Learning rate decay
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # scheduler = LR_Scheduler(
    #     optimizer,
    #     0, 0,
    #     100, 0.05 * bs / 256, 0,
    #     len(dataloaders['train']),
    # )

    # Model training routine
    print("\nTraining:-\n")

    # def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Feature list
        feature_save = []

        # Each epoch has a training and validation phase
        for phase in ['train']:  # TODO: orig: for phase in ['train', 'valid']:
            # if phase == 'train':
            #     model.train()  # Set model to training mode
            # else:
            #     model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            pred = []
            true = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:  # TODO: orig: for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # zero the parameter gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, feature = model(inputs)
                    feature_save.append(feature.cpu().data.numpy())
                    _, preds = torch.max(outputs, 1)
                    # loss = criterion(outputs, labels)
                    loss = criterion(outputs, labels)

                    preds_list = list(np.array(preds.cpu()))
                    labels_list = list(np.array(labels.cpu()))
                    pred.append(preds_list)
                    true.append(labels_list)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            pred = sum(pred, [])
            true = sum(true, [])
            f1 = f1_score(true, pred, average='macro')
            balance_acc = balanced_accuracy_score(true, pred)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.add_scalar('Train/Balance_accuracy', balance_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.add_scalar('Valid/Balance_accuracy', balance_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and balance_acc > best_acc:
                best_epoch = epoch
                best_acc = balance_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        # PATH = 'checkpoint/P16-7404;S6;UVM_R4_model/' + val_WSI_list + '/' + 'train_0622_' + val_WSI_list + '_' + str(epoch) + '.pth'
        feature_path = 'data_root/feature/P16-7404;S6;UVM_R4/' + 'train_0624_epoch_' + str(epoch)
        # torch.save(model, PATH)
        feature_save = np.squeeze(feature_save, axis=1)
        np.save(feature_path, feature_save)

    time_elapsed = time.time() - since


    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # print('Best epoch in val is : ', best_epoch)
    # train_info.append([val_WSI_list, best_epoch, best_acc])
# with open('checkpoint/P16-7404;S6;UVM_R4_info/train_info_0622_100percent.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerows(train_info)

'''
Sample run: python train.py --mode=finetune
'''
