import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image

orig_dir = 'data_root/learning/testing'
des_dir = 'data_root/learning/testing_contrast'

transform = iaa.Sequential([
    iaa.contrast.LinearContrast((1.0, 1.5))
])

for folder in os.listdir(orig_dir):
    for label_folder in os.listdir(f'{orig_dir}/{folder}'):
        for file in os.listdir(f'{orig_dir}/{folder}/{label_folder}'):
            path = f'{orig_dir}/{folder}/{label_folder}/{file}'
            img = imageio.imread(path)
            img_aug = transform(images=img)
            img_save = Image.fromarray(img_aug)
            save_dir = f'{des_dir}/{folder}/{label_folder}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_save.save(f'{save_dir}/{file}')
        print('.', end='')
