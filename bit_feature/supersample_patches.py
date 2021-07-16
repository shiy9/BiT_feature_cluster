import numpy as np
import openslide
import os
from PIL import Image

WSI_name = 'P18-8264;S2;UVM'
reg_num = 0  # reg_num is 0 based!
downsample = 2
patch_size = 256
draw_result_mask = True

# desired number of patches to supersample
target_num = 30

label_mask_path = f'data_root/tiles/{WSI_name}_R{reg_num}_labeled_tiles/label_mask/'
supersample_tile_path = f'data_root/tiles/{WSI_name}_R{reg_num}_ss_tiles/'
WSI_path = f'/home/yuxuanshi/VUSRP/WSI/'

slide = openslide.open_slide(f'{WSI_path}{WSI_name}.scn')
reg_x = int(slide.properties[f'openslide.region[{reg_num}].x'])
reg_y = int(slide.properties[f'openslide.region[{reg_num}].y'])
reg_w = int(slide.properties[f'openslide.region[{reg_num}].width'])
reg_h = int(slide.properties[f'openslide.region[{reg_num}].height'])

for mask_filename in os.listdir(label_mask_path):
    mask = np.load(label_mask_path + mask_filename)

    if draw_result_mask:
        result_mask = mask.copy()

    # randomly select a bunch of coordinates to serve as center of patches
    all_coords = np.where(mask > 0)
    row_coord = all_coords[0]
    col_coord = all_coords[1]
    coord_size = row_coord.size
    list_of_coord_idx = np.random.default_rng().choice(coord_size, size=target_num, replace=False)

    # Extract the patches from WSI
    label = mask_filename[:-9]
    label_tile_path = f'{supersample_tile_path}{label}/'
    if not os.path.exists(label_tile_path):
        os.makedirs(label_tile_path)

    for idx in list_of_coord_idx:
        assert mask[row_coord[idx]][col_coord[idx]] > 0, 'Selected non-255 region of mask'
        # relative coordinates at 40x
        patch_mid_x = col_coord[idx] * downsample
        patch_mid_y = row_coord[idx] * downsample
        patch_start_x = patch_mid_x - (patch_size * downsample / 2 - 1)
        patch_end_x = patch_mid_x + (patch_size * downsample / 2)
        patch_start_y = patch_mid_y - (patch_size * downsample / 2 - 1)
        patch_end_y = patch_mid_y + (patch_size * downsample / 2)

        # if patches will go out of bounds of entire region, reselect
        if patch_start_x < 0 or patch_start_y < 0 or patch_end_x >= mask.shape[1] or patch_end_y >= mask.shape[0]:
            replaced = False
            while not replaced:
                replaced_coord_idx = np.random.default_rng().choice(coord_size, size=1, replace=False)
                if replaced_coord_idx not in list_of_coord_idx:
                    replaced = True
            patch_mid_x = col_coord[replaced_coord_idx] * downsample
            patch_mid_y = row_coord[replaced_coord_idx] * downsample
            patch_start_x = patch_mid_x - (patch_size * downsample / 2 - 1)
            patch_end_x = patch_mid_x + (patch_size * downsample / 2)
            patch_start_y = patch_mid_y - (patch_size * downsample / 2 - 1)
            patch_end_y = patch_mid_y + (patch_size * downsample / 2)

        if draw_result_mask:
            result_mask[patch_mid_x//downsample][patch_mid_y//downsample] = 200
            temp = result_mask[patch_start_y//downsample: patch_start_y//downsample + patch_size,
            patch_start_x//downsample: patch_start_x//downsample + patch_size].copy()
            temp = np.where(temp != 200, 150, temp)
            result_mask[patch_start_y // downsample: patch_start_y // downsample + patch_size,
            patch_start_x // downsample: patch_start_x // downsample + patch_size] = temp

        # Calculate the absolute coordinates at 40x
        patch_start_x += reg_x
        patch_start_y += reg_y

        # Grab the patch
        naming_ct = 0
        patch = slide.read_region((patch_start_x, patch_start_y), 0, (patch_size * downsample, patch_size * downsample))
        patch = patch.resize((patch_size, patch_size))
        patch = patch.convert('RGB')
        patch.save(f'{label_tile_path}{WSI_name}_{label}_ss_{naming_ct}.png')
    if draw_result_mask:
        result_mask_path = f'{supersample_tile_path}verify_mask/'
        if not os.path.exists(result_mask_path):
            os.makedirs(result_mask_path)
        Image.fromarray(result_mask).save(f'{result_mask_path}/{WSI_name}_{label}_verify_mask.png')