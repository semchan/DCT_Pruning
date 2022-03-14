import numpy as np
import matplotlib.pyplot as plt
import os

def fnorm(arr):
    vmax = np.max(arr)
    vmin = np.min(arr)
    return (arr-vmin)/(vmax-vmin)

file_path = 'importance_score/vgg_16_bn_limit5'
dct = np.load(os.path.join(file_path, 'imp_conv_10.npy'))
# dct = np.load(os.path.join(file_path, 'imp_norm_10.npy'))
# dct = np.load(os.path.join(file_path, 'imp_relu_10.npy'))
rank = np.load(os.path.join(file_path, 'rank_conv_10.npy'))
# rank = np.load(os.path.join(file_path, 'rank_norm_10.npy'))
# rank = np.load(os.path.join(file_path, 'rank_relu_10.npy'))

dct=fnorm(dct)
rank=fnorm(rank)

plt.plot(dct, label='dct')
plt.plot(rank, label='rank')

plt.legend(loc='best', frameon=False, prop = {'size':8})

plt.show()