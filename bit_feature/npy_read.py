import numpy as np

# data = np.load('checkpoint/P16-7404;S6;UVM_R4_feature/test_R4/train_0622_test_R4_99.npy')
# data1 = np.load('checkpoint/P16-7404;S6;UVM_R4_feature/test_R4/train_0622_test_R4_0.npy')
# data = np.load('data_root/feature/P16-7404;S6;UVM_R4_feature.npy')
a = np.array([[0, 255], [255, 0]])
b = np.array([[255, 0], [0, 255]])
a = np.where(b > 0, 255, a)
print(a)
print('dummy print')