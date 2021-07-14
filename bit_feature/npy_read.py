import numpy as np
import openslide
import cv2

# Crop out interested region. For 7/13/21 Presentation
data = np.load('data_root/tiles_coord/P16-8917;S6;UVM_R0_tiles_coord.npy')
coord = data[32]
coord[0] = coord[0] - 512
coord[1] = coord[1] - 512
wsi = openslide.open_slide('/home/yuxuanshi/VUSRP/WSI/Processed/P16-8917;S6;UVM.scn')
size = 6 * 512
region = wsi.read_region(coord, 0, (size, size))
region = np.array(region.convert('RGB'))
cv2.imwrite('ReportImgs/temp_scrshot.png', region)



print('dummy print')