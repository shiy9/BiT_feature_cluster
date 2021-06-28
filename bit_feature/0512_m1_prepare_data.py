import numpy as np
import os
# import openslide
from PIL import Image
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
import random

# data_root = 'data_root'
# img_save_path = '/share/contrastive_learning/data/seg_on_79_5cluster_tumor_meeting/'
# img_patch_root = 'checkpoint/P16-7404;S6;UVM_R4_clustering'
# data_info_root = '/share/MIL/data/data_0408/'
# data_info_root = 'input/P16-7404;S6;UVM_R4_feature'
# data_info_root = '/share/MIL/data/all_data_simsiam_0601/'

# train_img_file = 'TCGA-3N-A9WC-01Z-00-DX1.C833FCAB-6329-4F90-88E5-CFDA0948047B.svs'
# train_img_file = 'TCGA-3N-A9WD-01Z-00-DX1.3B836595-3D67-4985-9D3B-39A7AE38B550.svs'

# WSI_list = os.listdir(data_root)

# rep_tumor_patch_list = []
# tumor_feature_arr = np.array([], dtype=np.int64).reshape(0, 2048)

WSI_name = 'P16-7404;S6;UVM'
reg_num = '4'

for _ in range(1):  # TODO: original for train_img_file in WSI_list:
    # print(train_img_file)
    # file_name = train_img_file

    # file_name_abbr = train_img_file.split('.')[0]
    fea_file = os.path.join('data_root/feature', f'{WSI_name}_R{reg_num}_feature.npy')
    # coor_file = os.path.join(data_info_root, 'coord', 'tiles_coord.npy')
    # label_file = os.path.join(data_info_root, 'label', file_name_abbr + '_label.npy')

    # coor_array = np.load(coor_file)
    # label_array = np.load(label_file)
    fea_array = np.load(fea_file)

    ### aggregate tumor patch for spectral clustering (7 is label of tumor)
    # tumor_index_list = np.where(label_array == 7)
    # print('tumor patch number:   ', len(tumor_index_list[0]))
    # tumor_index = list(tumor_index_list[0])
    # count = len(tumor_index)
    # if count > 500:
    #     sample_index = random.sample(tumor_index, 500)
    # else:
    #     sample_index = tumor_index

    ### build tumor feature array
    # fea_tumor = fea_array[sample_index, ...]
    clustering = SpectralClustering(n_clusters=5,
                 assign_labels="discretize",
                 random_state=0).fit(fea_array)  # Original: fea_tumor
    tumor_subclass_list = clustering.labels_
    # filename = ..._cluter_label
    np.save(f'data_root/cluster/{WSI_name}_R{reg_num}_cluster_label', tumor_subclass_list)

    # abstract top 5 patch in each cluster
    # for cluster_index in range(5):
    #     cluster0_index = np.where(tumor_subclass_list == cluster_index)
    #     cluster0_fea = fea_array[cluster0_index, ...][0]    # Original: fea_tumor
    #     if cluster0_fea.shape[0] >= 5:
    #         sim_matrix = 1 - pairwise_distances(cluster0_fea, metric="cosine")
    #         sim_sum = sim_matrix.sum(axis=1)
    #         topk = sim_sum.argsort()[-5:][::-1]
    #         for i in range(1):
    #             a = cluster0_index[0][topk[i]]
    #             index_in_all = a  # Original = tumor_index_list[0][a]
    #             patch_fea = fea_array[index_in_all]
    #             coor_x = coor_array[index_in_all][0]
    #             coor_y = coor_array[index_in_all][1]
    #             # img_patch_subfolder = train_img_file.split('.')[0]
    #             img_patch_name = os.path.join(img_patch_root, 'temp_'+str(coor_x)+'_'+str(coor_y)+'.0.png')
    #             rep_tumor_patch_list.append(img_patch_name)
    #             tumor_feature_arr = np.vstack((tumor_feature_arr, patch_fea))

# np.save('/share/MIL/data/data_0512_for_survival/feature_tumor_5cluster_rep_1.npy', tumor_feature_arr)
# np.save('checkpoint/P16-7404;S6;UVM_R4_clustering/cluster_fea/feature_5cluster_rep_1.npy', tumor_feature_arr)
# with open("checkpoint/P16-7404;S6;UVM_R4_clustering/cluster_fea/file_1.txt", 'w') as f:
#     for s in rep_tumor_patch_list:
#         f.write(str(s) + '\n')