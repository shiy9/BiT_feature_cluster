import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import cv2

orig_dir = 'data_root/learning/training'
des_dir = 'data_root/learning/training_contrast_cv'

# transform = iaa.Sequential([
#     iaa.contrast.LinearContrast((1.0, 1.5))
# ])

# # imgaug routine
# for folder in os.listdir(orig_dir):
#     for label_folder in os.listdir(f'{orig_dir}/{folder}'):
#         for file in os.listdir(f'{orig_dir}/{folder}/{label_folder}'):
#             path = f'{orig_dir}/{folder}/{label_folder}/{file}'
#             img = imageio.imread(path)
#             img_aug = transform(images=img)
#             img_save = Image.fromarray(img_aug)
#             save_dir = f'{des_dir}/{folder}/{label_folder}'
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             img_save.save(f'{save_dir}/{file}')
#         print('.', end='')

# opencv routine
for folder in os.listdir(orig_dir):
    for label_folder in os.listdir(f'{orig_dir}/{folder}'):
        for file in os.listdir(f'{orig_dir}/{folder}/{label_folder}'):
            path = f'{orig_dir}/{folder}/{label_folder}/{file}'
            img = cv2.imread(path)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            save_dir = f'{des_dir}/{folder}/{label_folder}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(f'{save_dir}/{file}', final)
        print('.', end='')
