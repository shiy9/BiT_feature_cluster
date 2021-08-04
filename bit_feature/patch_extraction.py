import numpy as np
import openslide
import os

# Extract a few patches
WSI_name = 'P17-4786;S6;UVM'
reg_num = 0
coord_file = np.load(f'data_root/tiles_coord/{WSI_name}_R{reg_num}_tiles_coord.npy')
coord_file_ls = list(coord_file)
neighbor_idx = [203, 203, 203, 203]
adjustments = [(512, -256), (3 * 256, -256), (512, 0), (3 * 256, 0)]

assert len(neighbor_idx) == len(adjustments), 'Two lists do not have same length!'

wsi = openslide.open_slide(f'/home/yuxuanshi/VUSRP/WSI/{WSI_name}.scn')
size = 256
fnames_now = sorted(os.listdir(f'data_root/tiles/{WSI_name}_R{reg_num}_tiles'))
save_num = int(fnames_now[-1].rsplit('_', 1)[-1][:-4])

for i in range(len(neighbor_idx)):
    coord = coord_file[neighbor_idx[i]]
    x_adj = coord[0] + adjustments[i][0]
    y_adj = coord[1] + adjustments[i][1]
    coord_file_ls.append(np.array([x_adj, y_adj]))
    patch = wsi.read_region((x_adj, y_adj), 0, (size, size))
    patch = patch.convert('RGB')
    save_num += 1
    patch.save(f'data_root/tiles/{WSI_name}_R{reg_num}_tiles/{WSI_name}_{save_num}.png')  # Note: change

np.save(f'data_root/tiles_coord/{WSI_name}_R{reg_num}_tiles_coord.npy', coord_file_ls)