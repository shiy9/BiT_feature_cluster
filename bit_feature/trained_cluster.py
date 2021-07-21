import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import time, os
import multiprocessing
import openslide
from PIL import Image
import BiT_models

test_tile_dir = 'data_root/tiles/Testing_10_WSI'
classifier_path = 'data_root/learning/best_model/train3_epoch_199.pth'

# Batch size
bs = 1
# Number of classes
num_classes = 6
# Number of workers
num_cpu = multiprocessing.cpu_count()

downsample = 2
patch_size = 256

image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transfer the model to GPU
model = BiT_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
model.load_from(np.load(f"{'BiT-M-R50x1'}.npz"))
classifier = torch.load(classifier_path)
model = model.to(device)
classifier = classifier.to(device)
model.eval()
classifier.eval()

# Model labels
classifier_labels = ['bzh', 'dis', 'eos', 'fibrotic lp', 'normal lp', 'others']
colors_list = [(201, 120, 25),  # light purple
               (123, 35, 15),  # brown
               (23, 172, 169),  # green
               (211, 49, 153),  # pink
                (160, 90, 160),  # darker pink/purple
               (200, 200, 200),  # gray
               (150, 200, 150), (200, 0, 0), (255, 255, 255)]

for folder in os.listdir(test_tile_dir):
    simg = openslide.open_slide(f'/home/yuxuanshi/VUSRP/WSI/{folder[:-9]}.scn')
    WSI_name = folder[:-6]
    reg_num = int(folder[-7])
    print(f'Processing {WSI_name}...', end='')
    max_w = int(simg.properties[f'openslide.region[{reg_num}].width'])  # matches x
    max_h = int(simg.properties[f'openslide.region[{reg_num}].height'])  # matches y
    img_x = int(simg.properties[f'openslide.region[{reg_num}].x'])
    img_y = int(simg.properties[f'openslide.region[{reg_num}].y'])

    img_x = int(int(np.floor(img_x / downsample / patch_size)) * downsample * patch_size)
    img_y = int(int(np.floor(img_y / downsample / patch_size)) * downsample * patch_size)
    max_w = int(int(np.ceil((img_x + max_w) / downsample / patch_size)) * downsample * patch_size) - img_x
    max_h = int(int(np.ceil((img_y + max_h) / downsample / patch_size)) * downsample * patch_size) - img_y

    dataset = datasets.ImageFolder(root=f'{test_tile_dir}/{folder}', transform=image_transforms['test'])
    dataset_len = len(dataset)
    dataloader = data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_cpu,
                                 pin_memory=True, drop_last=False)
    print('\tTesting-set size:', dataset_len, end='')

    coor_file_path = f'data_root/tiles_coord/Testing_10_WSI/{folder}_coord.npy'
    coor_array = np.load(coor_file_path)

    new_img_w = int(max_w / downsample // patch_size)
    new_img_h = int(max_h / downsample // patch_size)
    img_color = Image.new(mode="RGB", size=(new_img_w, new_img_h))

    img_color_arr = np.array(img_color).astype(np.uint8) + 255
    filename_list = sorted(os.listdir(f'{test_tile_dir}/{folder}/eoe'))
    for i, (inputs, _) in enumerate(dataloader, 0):
        inputs = inputs.to(device, non_blocking=True)
        spl_filename = dataloader.dataset.samples[i][0]
        spl_filename = spl_filename.rsplit('/', 1)[-1]
        assert spl_filename == filename_list[i], 'Coordinates and file do not match!'
        # forward
        _, feature = model(inputs)
        preds = classifier(feature)
        _, preds = torch.max(preds, 1)

        pred_label_idx = int(preds.data[0])

        coor_x = float(coor_array[i][0] - img_x) / downsample / patch_size
        coor_y = float(coor_array[i][1] - img_y) / downsample / patch_size

        color = colors_list[pred_label_idx]
        img_color_arr[int(coor_y)][int(coor_x)] = color

    print('\tSaving mask...', end='')
    Image.fromarray(img_color_arr).save(f'data_root/result_img/Testing_10_WSI/{WSI_name}_trained_mask.png')
    print('\tDone!')
