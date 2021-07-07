import openslide

WSI = openslide.open_slide('/home/yuxuanshi/VUSRP/WSI/P18-6324;S2;UVM.scn')
print(WSI.properties)