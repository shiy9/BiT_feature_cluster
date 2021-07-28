import numpy as np
import torch
# import torchvision
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
bs = 1
# Number of epochs
num_epochs = 100
# Note: Number of classes. Check before running
num_classes = 2
# Number of workers
num_cpu = multiprocessing.cpu_count()
# num_cpu = 0


for val_folder_index in range(1):  # Note: with validation: for val_folder_index in range(5):
    whole_train_list = ['folder1']
    # val_WSI_list = whole_train_list[val_folder_index]  # Note: valid cmt
    train_WSI_list = whole_train_list
    # train_WSI_list.pop(val_folder_index)  # Note: valid cmt

    train_directory = 'data_root/learning/training/'
    # Note: valid cmt
    # valid_directory = train_directory
    # Set the model save path
    # best_PATH = "models/train_beta.pth"

    # Applying transforms to the data
    image_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        # 'valid': transforms.Compose([
        #     transforms.Resize(size=256),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406],
        #                          [0.229, 0.224, 0.225])
        # ])
    }

    # Load data from folders
    dataset = {}
    dataset_train0 = datasets.ImageFolder(root=train_directory + train_WSI_list[0], transform=image_transforms['train'])
    # dataset_train1 = datasets.ImageFolder(root=train_directory + train_WSI_list[1], transform=image_transforms['train'])
    # dataset_train2 = datasets.ImageFolder(root=train_directory + train_WSI_list[2], transform=image_transforms['train'])
    # dataset_train3 = datasets.ImageFolder(root=train_directory + train_WSI_list[3], transform=image_transforms['train'])
    # dataset_train4 = datasets.ImageFolder(root=train_directory + train_WSI_list[4], transform=image_transforms['train'])
    # dataset_train5 = datasets.ImageFolder(root=train_directory + train_WSI_list[5], transform=image_transforms['train'])

    # Note: valid cmt
    # dataset['valid'] = datasets.ImageFolder(root=valid_directory + val_WSI_list, transform=image_transforms['valid'])

    # dataset['train'] = data.ConcatDataset([dataset_train0, dataset_train1, dataset_train2, dataset_train3, dataset_train4])
    dataset['train'] = dataset_train0

    # Size of train and validation data
    dataset_sizes = {
        'train': len(dataset['train']),
        # 'valid': len(dataset['valid'])  # Note: valid cmt
    }

    # Create iterators for data loading
    dataloaders = {
        'train': data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                                 num_workers=num_cpu, pin_memory=True, drop_last=True),
        # Note: valid cmt
        # 'valid': data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
        #                          num_workers=num_cpu, pin_memory=True, drop_last=True)
    }

    # Print the train and validation data sizes
    print("Training-set size:", dataset_sizes['train'])
          # "\nValidation-set size:", dataset_sizes['valid'])  # Note: valid cmt

    # Saving the classifier labels
    train_labels = dataset['train'].class_to_idx
    with open('data_root/learning/models/train_labels.txt', 'w') as train_labels_file:
        train_labels_file.write(json.dumps(train_labels))

    # Set default device as gpu, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    if train_mode == 'finetune':
        # Load a pretrained model - BiT
        print("\nLoading BiT-M-R50x1 for finetuning ...\n")
        model = BiT_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
        model.load_from(np.load(f"{'BiT-M-R50x1'}.npz"))
        classifier = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    elif train_mode == 'scratch':
        model_ft = models.resnet50(pretrained=False)
        fc_features = model_ft.fc.in_features
        # correct class number
        model_ft.fc = nn.Linear(fc_features, num_classes)
        # Set number of epochs to a higher value

    # Transfer the model to GPU
    model = model.to(device)
    classifier = classifier.to(device)
    # Print model summary
    print('Model Summary:-\n')

    # Loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(class_num=num_classes)

    # model = torch.nn.Sequential(model, classifier)

    # Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    # Note: may have problem
    # 0.001
    optimizer = optim.SGD(model.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)

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

    best_model_wts = copy.deepcopy(classifier.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        # Load the previous epoch's model in before training
        # For debugging only. Comment out later!
        if epoch != 0:
            classifier_lv = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
            classifier_lv.load_state_dict(torch.load(f'data_root/learning/models/train3_epoch_{epoch - 1}.pth'))
            classifier_lv = classifier_lv.to(device)
            running_corrects_v = 0
            running_corrects_lv = 0
            model.eval()
            classifier.eval()
            classifier_lv.eval()
            for inputs_v, labels_v in dataloaders['train']:
                inputs_v = inputs_v.to(device, non_blocking=True)
                labels_v = labels_v.to(device, non_blocking=True)

                # forward
                _, feature_v = model(inputs_v)
                preds_v = classifier(feature_v)
                preds_lv = classifier_lv(feature_v)
                _, preds_v = torch.max(preds_v, 1)
                _, preds_lv = torch.max(preds_lv, 1)

                running_corrects_v += torch.sum(preds_v == labels_v.data)
                running_corrects_lv += torch.sum(preds_lv == labels_v.data)

            epoch_acc_v = running_corrects_v.double() / dataset_sizes['train']
            epoch_acc_lv = running_corrects_lv.double() / dataset_sizes['train']
            print('Previous Epoch {} test Acc: {:4f}, load test Acc: {:4f}\n'.format(epoch - 1, epoch_acc_v, epoch_acc_lv))

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:  # Note: orig with valid: for phase in ['train', 'valid']:
            model.eval()    # Setting model to eval since we are only training the classifier
            classifier.train()

            # if phase == 'train':
            #     classifier.train()  # Set classifier to training mode
            # else:
            #     classifier.eval()  # Set classifier to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            pred = []
            true = []

            # Note: trouble shooting, delete later
            # Trying to see if the features are the same. Not correct with shuffle=True
            feature_dict = {}

            # Iterate over data.
            # Note: indexing does not really work with shuffle=True. Order of samples/imgs in dataloader is not right
            # Can definitely change to for inputs, labels in dataloaders[phase]
            for i, (inputs, labels) in enumerate(dataloaders[phase], 0):  # for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Note: trouble shooting, delete later
                spl_filename = dataloaders[phase].dataset.samples[i][0]
                spl_filename = spl_filename.rsplit('/', 1)[-1]

                # zero the parameter gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    _, feature = model(inputs)
                    classifier.zero_grad()
                    preds = classifier(feature)
                    # _, preds = torch.max(preds, 1)
                    loss = criterion(preds, labels)

                    # Note: trouble shooting, delete later
                    feature_dict[spl_filename] = feature.cpu().detach().numpy()

                    preds_list = list(np.array(preds.argmax(dim=1).cpu().detach().numpy()))
                    labels_list = list(np.array(labels.cpu().detach().numpy()))
                    pred.append(preds_list)
                    true.append(labels_list)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                temp = inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.argmax(dim=1) == labels.data)
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
            # if phase == 'valid' and balance_acc > best_acc:
            #     best_epoch = epoch
            #     best_acc = balance_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
        PATH = 'data_root/learning/models/train3_epoch_' + str(epoch) + '.pth'
        torch.save(classifier.state_dict(), PATH)

        # Note: trouble shooting, delete later
        np.save(f'data_root/learning/models/file_feature/train3_epoch_{str(epoch)}_fea.npy', feature_dict)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # print('Best val Acc: {:4f}'.format(best_acc))
    # print('Best epoch in val is : ', best_epoch)
    # train_info.append([val_WSI_list, best_epoch, best_acc])
writer.close()
# with open('data_root/learning/training_output/train2_summary.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerows(train_info)

'''
Sample run: python train.py --mode=finetue
'''
