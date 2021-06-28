import numpy as np
import os
import openslide
from PIL import Image

colors_list = [(201, 120, 25), (123, 35, 15), (23, 172, 169), (211, 49, 153),
                (160, 90, 160), (200, 200, 200),
               (150, 200, 150), (200, 0, 0), (255, 255, 255)]

data_root = 'data_root/'
# data_info_root = '/share/MIL/data/data_0408/'
# data_info_root = '/share/MIL/data/all_data_swav_0527/'
# img_save_path = '/share/MIL/data/seg_on_79_WSi_swav/'
# img_save_path = '/share/contrastive_learning/data/seg_on_79_WSI/'
# train_img_file = 'TCGA-3N-A9WC-01Z-00-DX1.C833FCAB-6329-4F90-88E5-CFDA0948047B'

WSI_list = os.listdir(data_root)

# Properties for the slide
patch_size = 256
reg_num = 4
downsample = 2
WSI_name = 'P16-7404;S6;UVM'
WSI_ext = '.scn'

for _ in range(1):  # Original: for train_img_file in WSI_list:
    # print(train_img_file)
    # file_name = train_img_file
    simg = openslide.open_slide(data_root + 'WSI/' + WSI_name + WSI_ext)
    max_w = int(simg.properties[f'openslide.region[{reg_num}].width'])  # matches x
    max_h = int(simg.properties[f'openslide.region[{reg_num}].height'])  # matches y
    img_x = int(simg.properties[f'openslide.region[{reg_num}].x'])
    img_y = int(simg.properties[f'openslide.region[{reg_num}].y'])

    img_x = int(int(np.floor(img_x / downsample / patch_size)) * downsample * patch_size)
    img_y = int(int(np.floor(img_y / downsample / patch_size)) * downsample * patch_size)
    max_w = int(int(np.ceil((img_x + max_w) / downsample / patch_size)) * downsample * patch_size) - img_x
    max_h = int(int(np.ceil((img_y + max_h) / downsample / patch_size)) * downsample * patch_size) - img_y


    print(f'{WSI_name}: Region {reg_num}    H: {max_h}  W: {max_w}')

    # file_name_abbr = train_img_file.split('.')[0]
    coor_file = os.path.join(data_root, 'tiles_coord', WSI_name + f'_R{reg_num}_tiles_coord.npy')
    label_file = os.path.join(data_root, 'cluster', WSI_name + f'_R{reg_num}_cluster_label.npy')

    new_img_w = int(max_w/downsample//patch_size)
    new_img_h = int(max_h/downsample//patch_size)
    img_color = Image.new(mode = "RGB", size = (new_img_w, new_img_h))

    img_color_arr = np.array(img_color).astype(np.uint8) + 255

    coor_array = np.load(coor_file)
    label_array = np.load(label_file)

    for index in range(coor_array.shape[0]):
        coor_x = float(coor_array[index][0] - img_x) / downsample / patch_size
        coor_y = float(coor_array[index][1] - img_y) / downsample / patch_size
        label = int(label_array[index])
        color = colors_list[label]
        img_color_arr[int(coor_y)][int(coor_x)] = color

    Image.fromarray(img_color_arr).save(f'{data_root}/result_img/{WSI_name}_R{reg_num}_cluster_mask.png')