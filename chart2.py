import numpy as np
import matplotlib.pyplot as plt
import os

file_path = 'importance_score/vgg_16_bn_limit5'
types = ['conv', 'norm', 'relu']

temp = []
imps = []
for t in types:
    for i in range(2):
        if i == 0:
            imp_path = os.path.join(file_path, 'rank_%s_0.npy'%t)
        else:
            imp_path = os.path.join(file_path, 'imp_%s_0.npy'%t)
        imp = np.load(imp_path)
        imp = (imp - min(imp))/(max(imp) - min(imp))
        temp.append(imp)
    imps.append(temp)
    temp = []

for idx, imp in enumerate(imps):
    plt.subplot(1,3,idx+1)
    if idx == 0:
        plt.plot(imp[0], label='rank')
        plt.plot(imp[1], label='dct')
        plt.legend(loc='best', frameon=False, prop = {'size':8})
        plt.title(types[idx])
    elif idx == 1:
        plt.plot(imp[0], label='rank')
        plt.plot(imp[1], label='dct')
        plt.legend(loc='best', frameon=False, prop = {'size':8})
        plt.title(types[idx])
    else:
        plt.plot(imp[0], label='rank')
        plt.plot(imp[1], label='dct')
        plt.legend(loc='best', frameon=False, prop = {'size':8})
        plt.title(types[idx])

plt.show()