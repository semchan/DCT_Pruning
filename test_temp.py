import argparse
import os,sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from thop import profile
import glob
from tqdm import tqdm
import numpy as np
from skimage import io
from PIL import Image
from utils.common import get_network, load_data, get_compress_rate
import time
from collections import OrderedDict

from data.data_loader import RescaleT
from data.data_loader import ToTensorLab
from data.data_loader import SalObjDataset

cudnn.benchmark = True
cudnn.enabled=True


def test():
    net.eval()
    cost_time = 0.
    with torch.no_grad():
        # data = torch.ones(args.batch_size, 3, 32, 32, device='cuda', dtype=torch.float32)
        data = torch.ones(args.batch_size, 3, 32, 32,dtype=torch.float32)
        if args.is_cpu:
            data = data.cpu()
            net.cpu()
        # torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(1000):
            net(data) 
        # torch.cuda.synchronize()
        cost_time += (time.time() - t0)
    print('\nBatch size:{}  Cost time: {:.2f}ms  Use_CPU: {}\n\n'.format(args.batch_size, cost_time*5, args.is_cpu))


if __name__ == '__main__':
# python test_temp.py --data_dir ./data --batch_size 64 --is_cpu --compress_rate [0.]*99 --net vgg_16_bn
    parser = argparse.ArgumentParser(description='Networks Pruning')
    parser.add_argument('--data_dir',type=str,default='./data',help='path to dataset')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('--is_cpu',default=True,action='store_true',help='use cpu')
    parser.add_argument('--compress_rate',type=str,default='[0.50]*7+[0.95]*5',help='compress rate of each conv')
    parser.add_argument(
        '--net',
        type=str,
        default='vgg_16_bn',
        choices=('resnet_50','vgg_16_bn','resnet_56',
                'resnet_110','densenet_40','googlenet'),
        help='net type')
    args = parser.parse_args()

    print('==> Building model..')
    compress_rate = get_compress_rate(args)
    net = get_network(args, compress_rate)
    # print('{}:'.format(args.net))

    # print('Compress_Rate: {}'.format(compress_rate))
    # flops, params = profile(net, inputs=(torch.randn(1, 3, 32, 32, 
    #                         device='cuda' if torch.cuda.is_available() else None),), verbose=False)
    # print('Params: %.2f' % (params))
    # print('Flops: %.2f' % (flops))

    test()
