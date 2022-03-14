import argparse
import os
import torch
from utils.common import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Importance Assessment')
    parser.add_argument('--dataset',type=str,default='cifar10',choices=('cifar10','imagenet','DUTS'),help='dataset')
    parser.add_argument('--data_dir',type=str,default='./data',help='path to dataset')
    parser.add_argument('--batch_size',type=int,default=128,help='batch size')
    parser.add_argument('--pretrain_dir',type=str,default='checkpoints/googlenet.pt',help='load the model from the specified checkpoint')
    parser.add_argument('--limit',type=int,default=5,help='The num of batch to get importence score.')
    parser.add_argument(
        '--net',
        type=str,
        default='googlenet',
        choices=('resnet_50','vgg_16_bn','resnet_56',
                'resnet_110','densenet_40','googlenet','u2netp'),
        help='net type')
    args = parser.parse_args()


    net = get_network(args)
    if args.pretrain_dir:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        if args.net=='u2netp':
            from collections import OrderedDict
            pretrained_dict = torch.load(args.pretrain_dir, map_location='cpu')
            new_state_dirct = OrderedDict()
            model_dict = net.state_dict()
            for k,v in pretrained_dict.items():
                new_state_dirct[k] = v
            pretrained_dict_current = {k: v for k, v in new_state_dirct.items() if k in model_dict}
            model_dict.update(pretrained_dict_current)
            net.load_state_dict(model_dict)
        else:
            if args.net=='vgg_16_bn' or args.net=='resnet_56':
                checkpoint = torch.load(args.pretrain_dir, map_location='cuda:0')
            else:
                checkpoint = torch.load(args.pretrain_dir)
            if args.net=='resnet_50':
                net.load_state_dict(checkpoint)
            elif args.net=='densenet_40' or args.net=='resnet_110':
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k.replace('module.', '')] = v
                net.load_state_dict(new_state_dict)
            else:
                net.load_state_dict(checkpoint['state_dict'])
        print('Completed! ')
    else:
        print('please speicify a pretrain model ')
        raise NotImplementedError

    # print(net)

    imp_score(net, args)
    # tmp(net, args)
