import json
import os
import shutil
import openslide
import cv2
import numpy as np
import PIL
from PIL import Image
import csv
from matplotlib import pyplot as plt
import sys

WSI_name = 'P17-4786;S6;UVM'
reg_num = 0  # reg_num is 0 based!
downsample = 2
patch_size = 256

# Note: change this if PyHIST is not usable
have_patches = True


json_root = '/home/yuxuanshi/VUSRP/big_transfer/bit_feature/data_root/annotation/converted_json/'
patch_root = f'/home/yuxuanshi/VUSRP/big_transfer/bit_feature/data_root/tiles/{WSI_name}_R{reg_num}_tiles/'
tiles_coord_dir = '/home/yuxuanshi/VUSRP/big_transfer/bit_feature/data_root/tiles_coord'
tiles_coord_path = f'/home/yuxuanshi/VUSRP/big_transfer/bit_feature/data_root/tiles_coord/{WSI_name}_R{reg_num}_tiles_coord.npy'
tiles_root = f'/home/yuxuanshi/VUSRP/big_transfer/bit_feature/data_root/tiles/{WSI_name}_R{reg_num}_tiles/'
save_root = f'/home/yuxuanshi/VUSRP/big_transfer/bit_feature/data_root/tiles/'
# train_list_csv_path = '/share/contrastive_learning/data/data_for_clean_512by512_step256/train.csv'
# train_list_csv = []

# train_img_file_list = ['TCGA-EB-A57M-01Z-00-DX1.FDE6CE11-9236-464B-ADDC-361FDD7CD8E6', \
#                        'TCGA-BF-AAP1-01Z-00-DX1.65888232-80E7-46F0-93D9-CFC6AE1D117E', \
#                        'TCGA-ER-A2NF-01Z-00-DX1.1468DD2D-6AC8-4657-A02E-E520668C676F', \
#                        'TCGA-D9-A4Z5-01Z-00-DX1.88AC8735-B520-4FCE-BC0B-AA6A611CE2D7', \
#                        'TCGA-EB-A431-01Z-00-DX1.9D1E5A19-63AE-48C4-B478-357DCCC1EB70', \
#                        'TCGA-EB-A550-01Z-00-DX1.10817DE0-BB6B-4539-9945-7B1B5B314C83', \
#                        'TCGA-EB-A299-01Z-00-DX1.22F1489B-6A55-411D-816B-C5DFD1A7BE52'
#                       ]


# file_list = os.listdir(root)

colors_list = [(201, 120, 25),  # light purple
               (123, 35, 15),  # brown
               (23, 172, 169),  # bluish
               (211, 49, 153),  # pink
                (160, 90, 160),  # darker pink/purple
               (200, 200, 200),  # gray
               (150, 200, 150), (201, 120, 25), (214, 199, 219), (42, 189, 89), (226, 2, 214), (200, 0, 0),
               (255, 255, 255)]

# TODO: official documentation is 'EA' instead of 'EI'
label_dict = {'eos': 2, 'bzh': 0, 'dis': 1, 'ea': 6, 'sl': 7, 'sea': 8, 'dec': 9, 'lpf': 10, 'normal lp': 4,
              'fibrotic lp': 3, 'ei': 11, 'others': 5}
pt_mask_dict = {'eos': 17, 'bzh': 34, 'dis': 51, 'ea': 68, 'sl': 85, 'sea': 102, 'dec': 119, 'lpf': 136,
                'normal lp': 153, 'fibrotic lp': 170, 'ei': 187}

if os.path.exists(f'{save_root}{WSI_name}_R{reg_num}_labeled_tiles/'):
    sys.exit('Sorted tiles folder already exists. Aborting...')

WSI_file_name = WSI_name + '.scn'
simg = openslide.open_slide(f'/home/yuxuanshi/VUSRP/WSI/{WSI_file_name}')

reg_x = int(simg.properties[f'openslide.region[{reg_num}].x'])
reg_y = int(simg.properties[f'openslide.region[{reg_num}].y'])
reg_w = int(simg.properties[f'openslide.region[{reg_num}].width'])
reg_h = int(simg.properties[f'openslide.region[{reg_num}].height'])

# Correction for entire WSI deep-zoom support, not needed?
# reg_x = int(int(np.floor(reg_x / downsample / patch_size)) * downsample * patch_size)
# reg_y = int(int(np.floor(reg_y / downsample / patch_size)) * downsample * patch_size)
# reg_w = int(int(np.ceil((reg_x + reg_w) / downsample / patch_size)) * downsample * patch_size) - reg_x
# max_h = int(int(np.ceil((reg_y + reg_h) / downsample / patch_size)) * downsample * patch_size) - reg_y

# TODO: not sure if this is needed
# file_name_front, _ = os.path.splitext(file_name)
# file_name_abbr = file_name_front[-4:]

# for file in file_list:
class_dict = {}
area_dict = {}
ratio_dict = {}

count = -1

print(f'{WSI_name} parsing annotation...', end='')
with open(f'{json_root}{WSI_name}_R{reg_num}.json') as json_file:
    data = json.load(json_file)
    for object in data:
        count += 1
        coor = object['geometry']['coordinates']
        if object['geometry']['type'] == 'Polygon':
            for coor_sub in coor:
                coor_arr = (np.array(coor_sub)/downsample).astype(int)
                try:
                    classification = object['properties']['classification']['name'].lower()
                    # print(classification)
                    if classification in class_dict:
                        canvas = class_dict[classification]
                    else:
                        canvas = np.zeros((reg_h//downsample, reg_w//downsample), dtype=np.uint8)
                        class_dict[classification] = canvas
                    cv2.fillPoly(canvas, [coor_arr], 255)
                    class_dict[classification] = canvas
                except Exception:
                    pass

        elif object['geometry']['type'] == 'LineString':
            coor_arr = (np.array(coor)/downsample).astype(int)
            try:
                classification = object['properties']['classification']['name'].lower()
                # print(classification)
                if classification in class_dict:
                    canvas = class_dict[classification]
                else:
                    canvas = np.zeros((reg_h//downsample, reg_w//downsample), dtype=np.uint8)
                    class_dict[classification] = canvas
                cv2.fillPoly(canvas, [coor_arr], 255)
                class_dict[classification] = canvas
            except Exception:
                pass

        elif object['geometry']['type'] == 'MultiPolygon':
            for polygon in coor:
                for coor_sub in polygon:
                    coor_arr = (np.array(coor_sub)/downsample).astype(int)
                    try:
                        classification = object['properties']['classification']['name'].lower()
                        # print(classification)
                        if classification in class_dict:
                            canvas = class_dict[classification]
                        else:
                            canvas = np.zeros((reg_h//downsample, reg_w//downsample), dtype=np.uint8)
                            class_dict[classification] = canvas
                        cv2.fillPoly(canvas, [coor_arr], 255)
                        class_dict[classification] = canvas
                    except Exception:
                        pass
print(' Done!')
sum_area = 0
for label in class_dict.keys():
    mask = class_dict[label]
    area = cv2.countNonZero(mask)

    if area > patch_size*patch_size:
        area_dict[label] = area
        sum_area += area

# save the binary mask to check annotation import
label_mask_dir = f'{save_root}{WSI_name}_R{reg_num}_labeled_tiles/label_mask/'

# Comment out this image part when not needed! Takes a long time!
# print('Saving label mask images...', end='')
# if not os.path.exists(f'{label_mask_dir}mask_image/'):
#     os.makedirs(f'{label_mask_dir}mask_image/')
# for label, mask in class_dict.items():
#     binary = mask > 0
#     plt.imsave(f'{label_mask_dir}mask_image/{WSI_name}_R{reg_num}_{label}_mask.png', binary)
# print(' Done!')
# for label in area_dict.keys():
#     ratio_dict[label] = area_dict[label] / sum_area

# save the label mask for "oversampling"
print('Saving label mask files...', end='')
if not os.path.exists(f'{label_mask_dir}mask_file/'):
    os.makedirs(f'{label_mask_dir}mask_file/')
for label, mask in class_dict.items():
    np.save(f'{label_mask_dir}mask_file/{label}_mask.npy', mask)
print(' Done!')

# For drawing mask, new_img_w and new_img_h in terms of number of patches
new_img_w = int(reg_w/downsample//patch_size)
new_img_h = int(reg_h/downsample//patch_size)
img_color = Image.new(mode = "RGB", size = (new_img_w, new_img_h))
img_color_arr = np.array(img_color).astype(np.uint8) + 255

if have_patches:
    tiles_coord = np.load(tiles_coord_path)
    i = 0

    # Write a CSV file for record
    print('Sorting patches...', end='')
    with open(f'{save_root}{WSI_name}_R{reg_num}_labeled_tiles/{WSI_name}_R{reg_num}_tiles_record.csv', mode='w') as tiles_record:
        fieldnames = ['patch_name', 'eos', 'bzh', 'dis', 'ea', 'sl', 'sea', 'dec', 'lpf', 'normal lp', 'fibrotic lp', 'ei',
                      'supersampled']
        writer = csv.DictWriter(tiles_record, fieldnames=fieldnames)
        writer.writeheader()

        for filename in sorted(os.listdir(tiles_root)):
            # print(f'File being processed: {filename}')
            # at 40x resolution and are relative coordinates in terms of region
            csv_row = {'patch_name': filename, 'eos': 0, 'bzh': 0, 'dis': 0, 'ea': 0, 'sl': 0, 'sea': 0, 'dec': 0, 'lpf': 0,
                       'normal lp': 0, 'fibrotic lp': 0, 'ei': 0, 'supersampled': 0}
            patch_start_x = tiles_coord[i][0] - reg_x
            patch_start_y = tiles_coord[i][1] - reg_y
            saved = False

            # Not used, center_x, y are relative coordinates at 40x resolution
            # center_x = patch_start_x + patch_size * downsample / 2
            # center_y = patch_start_y + patch_size * downsample / 2

            first_mask_area = 0
            second_mask_area = 0
            third_mask_area = 0
            first_label = '_'
            second_label = '_'
            third_label = '_'

            # The grayscale mask of each patch containing all the labels within that patch
            patch_save_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

            for label in class_dict.keys():
                subject_mask = class_dict[label]
                patch_mask = subject_mask[patch_start_y // downsample: patch_start_y // downsample + patch_size,
                             patch_start_x // downsample: patch_start_x // downsample + patch_size]
                patch_save_mask = np.where(patch_mask > 0, pt_mask_dict[label], patch_save_mask)

                mask_area = cv2.countNonZero(patch_mask)
                if mask_area > first_mask_area:
                    third_mask_area = second_mask_area
                    third_label = second_label
                    second_mask_area = first_mask_area
                    second_label = first_label
                    first_mask_area = mask_area
                    first_label = label
                elif mask_area > second_mask_area:
                    third_mask_area = second_mask_area
                    third_label = second_label
                    second_mask_area = mask_area
                    second_label = label
                elif mask_area > third_mask_area:
                    third_mask_area = mask_area
                    third_label = label

                # V2
                if mask_area > 0 and (label == 'dis' or label == 'eos'):
                    sorted_root = f'{save_root}{WSI_name}_R{reg_num}_labeled_tiles/{label}'
                    if not os.path.exists(sorted_root):
                        os.makedirs(sorted_root)
                    shutil.copy(f'{tiles_root}{filename}', f'{sorted_root}/{filename}')
                    csv_row[label] = 1
                    saved = True

            temp_x = int(patch_start_x / downsample // patch_size)
            temp_y = int(patch_start_y / downsample // patch_size)

            others_root = f'{save_root}{WSI_name}_R{reg_num}_labeled_tiles/others'
            if not os.path.exists(others_root):
                os.makedirs(others_root)

            if first_label == '_':
                img_color_arr[temp_y, temp_x] = colors_list[label_dict['others']]
                shutil.copy(f'{tiles_root}{filename}', f'{others_root}/{filename}')
            else:
                writer.writerow(csv_row)
                color_index = label_dict[first_label]
                img_color_arr[temp_y, temp_x] = colors_list[color_index]

                sorted_mask_root = f'{save_root}{WSI_name}_R{reg_num}_labeled_tiles/tiles_mask'
                sorted_mask_file_root = sorted_mask_root + '/mask_file'
                sorted_mask_img_root = sorted_mask_root + '/mask_img'

                if not os.path.exists(sorted_mask_root):  # V1: sorted root
                    os.makedirs(sorted_mask_root)
                if not os.path.exists(sorted_mask_file_root):
                    os.makedirs(sorted_mask_file_root)
                if not os.path.exists(sorted_mask_img_root):
                    os.makedirs(sorted_mask_img_root)

                if not saved:
                    sorted_root = f'{save_root}{WSI_name}_R{reg_num}_labeled_tiles/{first_label}'
                    if not os.path.exists(sorted_root):
                        os.makedirs(sorted_root)
                    shutil.copy(f'{tiles_root}{filename}', f'{sorted_root}/{filename}')
                    csv_row[first_label] = 1

                # V2
                np.save(f'{sorted_mask_file_root}/{filename}_mask.npy', patch_save_mask)
                Image.fromarray(patch_save_mask).save(f'{sorted_mask_img_root}/{filename}_mask.png')

            # print(temp_x, temp_y, first_label)
            i += 1

        tiles_record.close()
        shutil.move(tiles_root, f'{save_root}Unlabeled')
        Image.fromarray(img_color_arr).save(f'data_root/result_img/{WSI_name}_R{reg_num}_patch_group_mask.png')
    print(' Done!')

else:
    # Crop image
    patch_h = 512
    patch_w = 512
    step = 512

    crop_x_total = reg_w // step - 1
    crop_y_total = reg_h // step - 1
    coord_list = []
    print('Cropping image...', end='')
    with open(f'{save_root}{WSI_name}_R{reg_num}_labeled_tiles/{WSI_name}_R{reg_num}_tiles_record.csv', mode='w') as tiles_record:
        fieldnames = ['patch_name', 'eos', 'bzh', 'dis', 'ea', 'sl', 'sea', 'dec', 'lpf', 'normal lp', 'fibrotic lp', 'ei',
                      'supersampled']
        writer = csv.DictWriter(tiles_record, fieldnames=fieldnames)
        writer.writeheader()

        # For saving file names
        save_ctr = 0

        # Note: NOT saving the grayscale mask
        for j in range(crop_y_total):
            for i in range(crop_x_total):
                filename = f'{WSI_name}_{save_ctr}'
                csv_row = {'patch_name': filename, 'eos': 0, 'bzh': 0, 'dis': 0, 'ea': 0, 'sl': 0, 'sea': 0, 'dec': 0,
                           'lpf': 0, 'normal lp': 0, 'fibrotic lp': 0, 'ei': 0, 'supersampled': 0}
                # Absolute coordinates at 40x resolution
                patch_start_x = i * step + reg_x
                patch_start_y = j * step + reg_y

                # Relative coordinates at 40x resolution
                patch_start_x_rl = patch_start_x - reg_x
                patch_start_y_rl = patch_start_y - reg_y

                first_mask_area = 0
                second_mask_area = 0
                third_mask_area = 0
                first_label = '_'
                second_label = '_'
                third_label = '_'

                for label in class_dict.keys():
                    subject_mask = class_dict[label]
                    patch_mask = subject_mask[patch_start_y_rl//downsample: patch_start_y_rl//downsample + patch_size,
                                 patch_start_x_rl//downsample: patch_start_x_rl//downsample + patch_size]
                    mask_area = cv2.countNonZero(patch_mask)
                    if mask_area > first_mask_area:
                        third_mask_area = second_mask_area
                        third_label = second_label
                        second_mask_area = first_mask_area
                        second_label = first_label
                        first_mask_area = mask_area
                        first_label = label
                    elif mask_area > second_mask_area:
                        third_mask_area = second_mask_area
                        third_label = second_label
                        second_mask_area = mask_area
                        second_label = label
                    elif mask_area > third_mask_area:
                        third_mask_area = mask_area
                        third_label = label

                    if mask_area > 0:
                        img_patch = simg.read_region((patch_start_x, patch_start_y), 0, (512, 512)).convert('RGB')
                        img_patch = img_patch.resize((patch_size, patch_size), resample=PIL.Image.BICUBIC)
                        sorted_root = f'{save_root}{WSI_name}_R{reg_num}_labeled_tiles/{label}'
                        if not os.path.exists(sorted_root):
                            os.makedirs(sorted_root)
                        img_patch = img_patch.convert('RGB')
                        img_patch.save(f'{sorted_root}/{filename}.png')
                        csv_row[label] = 1
                        coord_list.append([patch_start_x, patch_start_y])
                if first_label == '_':
                    img_color_arr[j, i] = colors_list[0]
                else:
                    color_index = label_dict[first_label]
                    img_color_arr[j, i] = colors_list[color_index]
                    writer.writerow(csv_row)
                    save_ctr += 1
        np.save(f'{tiles_coord_dir}/{WSI_name}_R{reg_num}_tiles_coord.npy', coord_list)
    print('\tDone!')
