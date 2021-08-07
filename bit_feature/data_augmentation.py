import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import cv2
import numpy as np
import random

orig_dir = 'data_root/learning/testing'
des_dir = 'data_root/learning/testing_flip_sharp'

if os.path.exists(des_dir):
    if len(os.listdir(des_dir)) != 0:
        input('Destination folder not empty. Press Enter to continue...')

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
# increase contrast
# for folder in os.listdir(orig_dir):
#     for label_folder in os.listdir(f'{orig_dir}/{folder}'):
#         for file in os.listdir(f'{orig_dir}/{folder}/{label_folder}'):
#             path = f'{orig_dir}/{folder}/{label_folder}/{file}'
#             img = cv2.imread(path)
#             lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#             l, a, b = cv2.split(lab)
#             clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
#             cl = clahe.apply(l)
#             limg = cv2.merge((cl, a, b))
#             final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#             save_dir = f'{des_dir}/{folder}/{label_folder}'
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             cv2.imwrite(f'{save_dir}/{file}', final)
#         print('.', end='')


# opencv routine
# increase contrast and rotate
# for folder in os.listdir(orig_dir):
#     for label_folder in os.listdir(f'{orig_dir}/{folder}'):
#         for file in os.listdir(f'{orig_dir}/{folder}/{label_folder}'):
#             path = f'{orig_dir}/{folder}/{label_folder}/{file}'
#             img = cv2.imread(path)
#             lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#             l, a, b = cv2.split(lab)
#             clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
#             cl = clahe.apply(l)
#             limg = cv2.merge((cl, a, b))
#             ctrst = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#             image_center = tuple(np.array(ctrst.shape[1::-1]) / 2)
#             angle = random.random() * 360
#             rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#             rot_res = cv2.warpAffine(ctrst, rot_mat, ctrst.shape[1::-1], flags=cv2.INTER_LINEAR)
#             save_dir = f'{des_dir}/{folder}/{label_folder}'
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             cv2.imwrite(f'{save_dir}/{file}', rot_res)
#         print('.', end='')


# opencv routine
# increase contrast and random flip
# for folder in os.listdir(orig_dir):
#     for label_folder in os.listdir(f'{orig_dir}/{folder}'):
#         for file in os.listdir(f'{orig_dir}/{folder}/{label_folder}'):
#             path = f'{orig_dir}/{folder}/{label_folder}/{file}'
#             img = cv2.imread(path)
#             lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#             l, a, b = cv2.split(lab)
#             clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
#             cl = clahe.apply(l)
#             limg = cv2.merge((cl, a, b))
#             ctrst = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#             flip_code = random.randint(-1, 1)
#             need_flip = bool(random.getrandbits(1))
#             if need_flip:
#                 res = cv2.flip(ctrst, flip_code)
#             else:
#                 res = ctrst
#             save_dir = f'{des_dir}/{folder}/{label_folder}'
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             cv2.imwrite(f'{save_dir}/{file}', res)
#         print('.', end='')

# opencv routine
# increase contrast + sharpen + Gaussian noise
# for folder in os.listdir(orig_dir):
#     for label_folder in os.listdir(f'{orig_dir}/{folder}'):
#         for file in os.listdir(f'{orig_dir}/{folder}/{label_folder}'):
#             path = f'{orig_dir}/{folder}/{label_folder}/{file}'
#             img = cv2.imread(path)
#             lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#             l, a, b = cv2.split(lab)
#             clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
#             cl = clahe.apply(l)
#             limg = cv2.merge((cl, a, b))
#             ctrst = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#             kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#             sharpen = cv2.filter2D(ctrst, -1, kernel)
#             row, col, ch = sharpen.shape
#             mean = 0
#             var = 0.1
#             sigma = var ** 0.5
#             gauss = np.random.normal(mean, sigma, (row, col, ch))
#             gauss = gauss.reshape(row, col, ch)
#             noisy = sharpen + gauss
#             noisy = noisy.astype('uint8')
#             save_dir = f'{des_dir}/{folder}/{label_folder}'
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             cv2.imwrite(f'{save_dir}/{file}', noisy)
#         print('.', end='')


# # opencv routine
# # increase contrast + flip + rot + sharpen
# for folder in os.listdir(orig_dir):
#     for label_folder in os.listdir(f'{orig_dir}/{folder}'):
#         for file in os.listdir(f'{orig_dir}/{folder}/{label_folder}'):
#             path = f'{orig_dir}/{folder}/{label_folder}/{file}'
#             img = cv2.imread(path)
#             lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#             l, a, b = cv2.split(lab)
#             clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
#             cl = clahe.apply(l)
#             limg = cv2.merge((cl, a, b))
#             ctrst = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#
#             #####
#             flip_code = random.randint(-1, 1)
#             need_flip = bool(random.getrandbits(1))
#             if need_flip:
#                 res = cv2.flip(ctrst, flip_code)
#             else:
#                 res = ctrst
#
#             #####
#             kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#             sharpen = cv2.filter2D(res, -1, kernel)
#
#             save_dir = f'{des_dir}/{folder}/{label_folder}'
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             cv2.imwrite(f'{save_dir}/{file}', sharpen)
#         print('.', end='')


# opencv routine
# increase contrast + flip + rot + sharpen
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
            ctrst = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            #####
            flip_code = random.randint(-1, 1)
            need_flip = bool(random.getrandbits(1))
            if need_flip:
                res = cv2.flip(ctrst, flip_code)
            else:
                res = ctrst

            #####
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpen = cv2.filter2D(res, -1, kernel)
            save_dir = f'{des_dir}/{folder}/{label_folder}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(f'{save_dir}/{file}', sharpen)
        print('.', end='')
