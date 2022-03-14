import numpy as np
import matplotlib.pyplot as plt
import os

conv_num = 110

# 'resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet','u2netp'

file_path = []
file_path.append('importance_score/resnet_110')
file_path.append('importance_score/resnet_110_limit5')

temp = []
imps = []
for idx, path in enumerate(file_path):
    for i in range(1, conv_num):
        if idx == 0:
            imp_path = os.path.join(path, 'rank_conv%d.npy'%i)
        else:
            imp_path = os.path.join(path, 'imp_conv%d.npy'%i)
        imp = np.load(imp_path)
        temp.append(imp.sum())
    temp = (temp - min(temp))/(max(temp) - min(temp))
    imps.append(temp)
    temp = []

for idx, imp in enumerate(imps):
    if idx == 0:
        plt.plot(imp, label='rank')
    else:
        plt.plot(imp, label='dct')

plt.title('resnet_110')
plt.legend(loc='best', frameon=False, prop = {'size':8})
plt.show()