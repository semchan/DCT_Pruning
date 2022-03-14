import torch
import torch.utils
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import cv2
import sys,os
import numpy as np
import math
import re

from data.data_loader import RescaleT
from data.data_loader import RandomCrop
from data.data_loader import ToTensorLab
from data.data_loader import SalObjDataset

from models.cifar10.vgg import vgg_16_bn
from models.cifar10.resnet import resnet_56,resnet_110
from models.cifar10.googlenet import googlenet
from models.cifar10.densenet import densenet_40
from models.imagenet.resnet import resnet_50
from models.DUTS.u2net import U2NETP as u2netp


import torch_dct as dct


def get_network(args, compress_rate=[0.]*100):

    if args.net == 'vgg_16_bn':
        net = vgg_16_bn(compress_rate=compress_rate)
    elif args.net == 'resnet_50':
        net = resnet_50(compress_rate=compress_rate)
    elif args.net == 'resnet_110':
        net = resnet_110(compress_rate=compress_rate)
    elif args.net == 'googlenet':
        net = googlenet(compress_rate=compress_rate)
    elif args.net == 'densenet_40':
        net = densenet_40(compress_rate=compress_rate)
    elif args.net == 'resnet_56':
        net = resnet_56(compress_rate=compress_rate)
    elif args.net == 'u2netp':
        net = u2netp(compress_rate=compress_rate)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def load_data(args):

    # load training data
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                                transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    elif args.dataset == 'imagenet':
        pin_memory = True if torch.cuda.is_available() else False

        scale_size = 224

        traindir = os.path.join(args.data_dir, 'ILSVRC2012_img_train')
        valdir = os.path.join(args.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = DataLoader(trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=pin_memory)
        print('Files already downloaded and verified')

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        val_loader = DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True)
        print('Files already downloaded and verified')

    elif args.dataset == 'DUTS':
        import glob
        from torch.utils.data import Dataset

        data_dir = args.data_dir
        data_dir += '/' if not data_dir[-1]=='/' else ''
        tra_image_dir = os.path.join('DUTS-TR', 'DUTS-TR-Image' + os.sep)
        tra_label_dir = os.path.join('DUTS-TR', 'DUTS-TR-Mask' + os.sep)
        image_ext = '.jpg'
        label_ext = '.png'
        tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
        tra_lbl_name_list = []

        for img_path in tra_img_name_list:
            img_name = img_path.split(os.sep)[-1]

            aaa = img_name.split(".")
            bbb = aaa[0:-1]
            imidx = bbb[0]
            for i in range(1,len(bbb)):
                imidx = imidx + "." + bbb[i]

            tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

        print("---")
        print("train images: ", len(tra_img_name_list))
        print("train labels: ", len(tra_lbl_name_list))
        print("---")

        salobj_dataset = SalObjDataset(
            img_name_list=tra_img_name_list,
            lbl_name_list=tra_lbl_name_list,
            transform=transforms.Compose([
                RescaleT(320),
                RandomCrop(288),
                ToTensorLab(flag=0)]))
        train_loader = DataLoader(salobj_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

        return train_loader, None

    return train_loader, val_loader


def get_compress_rate(args):

    cprate_str = args.compress_rate
    cprate_str_list = cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*', ''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    return cprate


def adjust_learning_rate(optimizer, epoch, step, len_iter, args):

    if args.lr_type == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.learning_rate * (0.1 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.learning_rate * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.learning_rate * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.learning_rate
    else:
        raise NotImplementedError

    #Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    if step == 0:
        print('learning_rate: ' + str(lr))


#label smooth
class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def torch2dct(feature_map):

    t = feature_map.cpu().numpy()
    t = np.float32(t)
    # 
    if t.shape[0] % 2 != 0:
        t = np.pad(t, (1,0), 'constant')
    dct = cv2.dct(t)

    return torch.from_numpy(dct)


def norm(arr):
    vmin = np.min(arr)
    vmax = np.max(arr)
    arr = (arr - vmin)/(vmax - vmin)
    return arr


def cnt_score(dct_list):

    for idx, dct in enumerate(dct_list):
        dct_list[idx] = torch.sum(dct.mul(dct)).item()
    # norm(dct_list)

    return torch.tensor(dct_list)


feature_result = torch.tensor(0.)
total = torch.tensor(0.)


def get_feature_hook(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = [dct.dct_2d(output[i,j,:,:], norm='ortho') for i in range(a) for j in range(b)]
    # c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])
    # c = [dct.dct_3d(output[i,:,:,:], norm='ortho') for i in range(a)]

    c = cnt_score(c)

    c = c.view(a, -1)
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def get_feature_hook_densenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = [torch2dct(output[i,j,:,:]) for i in range(a) for j in range(b-12,b)]

    c = cnt_score(c)

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def get_feature_hook_u2net_input(self, input, output):
    global feature_result
    global total
    a = input[0].shape[0]
    b = input[0].shape[1]
    c = [torch2dct(input[0][i,j,:,:]) for i in range(a) for j in range(b)]

    c = cnt_score(c)

    c = c.view(a, -1)
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def inference(net, train_loader, limit):
    net.eval()
    for batch_idx, (data, _) in enumerate(train_loader):
        if batch_idx >= limit:
            break
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            net(data)


def u2netp_inference(net, train_loader, limit):
    net.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            if batch_idx >= limit:
               break
            inputs = data['image']
            inputs = inputs.type(torch.FloatTensor)
            inputs_v = Variable(inputs.cuda(), requires_grad=False)
            net(inputs_v)


def tmp(net, args):
    if not os.path.isdir('importance_score'):
        os.mkdir('importance_score')
    if not os.path.isdir('importance_score/' + args.net + '_limit' + str(args.limit)):
        os.mkdir('importance_score/' + args.net + '_limit' + str(args.limit))

    print('==> Loading data of {}..'.format(args.dataset))
    train_loader, _ = load_data(args)

    print('==> Generating importance score..')
    print('Importance Score is located at ./importance_score/' + args.net + '_limit' + str(args.limit))
    global feature_result
    global total


    if args.net=='vgg_16_bn':

        # cov_layer = net.features.conv10
        # cov_layer = net.features.norm10
        cov_layer = net.features.relu10
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference(net, train_loader, args.limit)
        handler.remove()

        np.save('importance_score/' + args.net + '_limit' + str(args.limit) + '/imp_relu_10.npy', feature_result.numpy())
        # np.save('importance_score/' + args.net + '_limit' + str(args.limit) + '/rank_relu_10.npy', feature_result.numpy())
        # print('/imp_conv' + str(i + 1) + ':done!')
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)



def imp_score(net, args):
    if not os.path.isdir('importance_score'):
        os.mkdir('importance_score')
    if not os.path.isdir('importance_score/' + args.net + '_limit' + str(args.limit)):
        os.mkdir('importance_score/' + args.net + '_limit' + str(args.limit))

    print('==> Loading data of {}..'.format(args.dataset))
    train_loader, _ = load_data(args)

    print('==> Generating importance score..')
    print('Importance Score is located at ./importance_score/' + args.net + '_limit' + str(args.limit))
    global feature_result
    global total
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)


    if args.net=='vgg_16_bn':

        relucfg = net.relucfg

        for i, cov_id in enumerate(relucfg):
            cov_layer = net.features[cov_id]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference(net, train_loader, args.limit)
            handler.remove()

            np.save('importance_score/' + args.net + '_limit' + str(args.limit) + '/imp_conv' + str(i + 1) + '.npy', feature_result.numpy())
            print('/imp_conv' + str(i + 1) + ':done!')
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)


    elif args.net=='resnet_56':

        cov_layer = eval('net.relu')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        net.eval()
        inference(net, train_loader, args.limit)
        handler.remove()

        np.save('importance_score/' + args.net + '_limit'+str(args.limit) + '/imp_conv1' + '.npy', feature_result.numpy())
        print('/imp_conv1' + ':done!')
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # ResNet56 per block
        cnt=1
        for i in range(3):
            block = eval('net.layer%d' % (i + 1))
            for j in range(9):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(cnt + 1) + '.npy', feature_result.numpy())
                print('/imp_conv%d'%(cnt + 1) + ':done!')
                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference(net, train_loader, args.limit)
                handler.remove()
                np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(cnt + 1) + '.npy', feature_result.numpy())
                print('/imp_conv%d'%(cnt + 1) + ':done!')
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)


    elif args.net=='densenet_40':

        # Densenet per block & transition
        for i in range(3):
            dense = eval('net.dense%d' % (i + 1))
            for j in range(12):
                cov_layer = dense[j].relu
                if j==0:
                    handler = cov_layer.register_forward_hook(get_feature_hook)
                else:
                    handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
                inference(net, train_loader, args.limit)
                handler.remove()
                np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(13*i+j+1) + '.npy', feature_result.numpy())
                print('/imp_conv%d'%(13*i+j+1) + ':done!')
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

            if i<2:
                trans=eval('net.trans%d' % (i + 1))
                cov_layer = trans.relu
                handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
                inference(net, train_loader, args.limit)
                handler.remove()
                np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(13 * (i+1)) + '.npy', feature_result.numpy())
                print('/imp_conv%d'%(13 * (i+1)) + ':done!')
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

        cov_layer = net.relu
        handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
        inference(net, train_loader, args.limit)
        handler.remove()
        np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(39) + '.npy', feature_result.numpy())
        print('/imp_conv%d'%(39) + ':done!')
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)


    elif args.net=='googlenet':

        cov_list=['pre_layers',
                'inception_a3',
                'maxpool1',
                'inception_a4',
                'inception_b4',
                'inception_c4',
                'inception_d4',
                'maxpool2',
                'inception_a5',
                'inception_b5',
                ]

        # branch type
        tp_list=['n1x1','n3x3','n5x5','pool_planes']
        for idx, cov in enumerate(cov_list):

            cov_layer=eval('net.'+cov)

            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference(net, train_loader, args.limit)
            handler.remove()

            if idx>0:
                for idx1,tp in enumerate(tp_list):
                    if idx1==3:
                        np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d_'%(idx+1) + tp + '.npy', 
                                feature_result[sum(net.filters_p[idx-1][:-1]) : sum(net.filters_p[idx-1][:])].numpy())
                        print('/imp_conv%d'%(idx+1) + tp + ':done!')
                    else:
                        np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d_'%(idx+1) + tp + '.npy', 
                                feature_result[sum(net.filters_p[idx-1][:idx1]) : sum(net.filters_p[idx-1][:idx1+1])].numpy())
                        print('/imp_conv%d'%(idx+1) + tp + ':done!')
            else:
                np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d_'%(idx+1) + '.npy', feature_result.numpy())
                print('/imp_conv%d'%(idx+1) + ':done!')
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)


    elif args.net=='resnet_110':

        cov_layer = eval('net.relu')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference(net, train_loader, args.limit)
        handler.remove()
        np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(1) + '.npy', feature_result.numpy())
        print('/imp_conv%d'%(1) + ':done!')
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cnt = 1
        # ResNet110 per block
        for i in range(3):
            block = eval('net.layer%d' % (i + 1))
            for j in range(18):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference(net, train_loader, args.limit)
                handler.remove()
                np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(cnt+1) + '.npy', feature_result.numpy())
                print('/imp_conv%d'%(cnt+1) + ':done!')
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference(net, train_loader, args.limit)
                handler.remove()
                np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(cnt+1) + '.npy', feature_result.numpy())
                print('/imp_conv%d'%(cnt+1) + ':done!')
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)


    elif args.net=='resnet_50':

        cov_layer = eval('net.maxpool')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference(net, train_loader, args.limit)
        handler.remove()
        np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(1) + '.npy', feature_result.numpy())
        print('/imp_conv%d'%(1) + ':done!')
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # ResNet50 per bottleneck
        cnt=1
        for i in range(4):
            block = eval('net.layer%d' % (i + 1))
            for j in range(net.num_blocks[i]):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference(net, train_loader, args.limit)
                handler.remove()
                np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(cnt+1) + '.npy', feature_result.numpy())
                print('/imp_conv%d'%(cnt+1) + ':done!')
                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference(net, train_loader, args.limit)
                handler.remove()
                np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(cnt+1) + '.npy', feature_result.numpy())
                print('/imp_conv%d'%(cnt+1) + ':done!')
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu3
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference(net, train_loader, args.limit)
                handler.remove()
                if j==0:
                    #shortcut conv
                    np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(cnt+1) + '.npy', feature_result.numpy())
                    print('/imp_conv%d'%(cnt+1) + ':done!')
                    cnt += 1
                #conv3
                np.save('importance_score/' + args.net+'_limit' + str(args.limit) + '/imp_conv%d'%(cnt+1) + '.npy', feature_result.numpy())
                print('/imp_conv%d'%(cnt+1) + ':done!')
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)


    elif args.net=='u2netp':

        for i in range(6):
            cov_layer_name ='net.stage%d.rebnconvin.relu_s1' % (i+1)
            print("current layer:",cov_layer_name)
            block = eval(cov_layer_name)
            cov_layer = block
            handler = cov_layer.register_forward_hook(get_feature_hook)
            u2netp_inference(net, train_loader, args.limit)
            handler.remove()

            np.save('importance_score/' + args.net +'_limit%d'%(args.limit) + '/' + cov_layer_name + '.npy', feature_result.numpy())
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            if i<5:
                cov_layer_name ='net.stage%dd.rebnconvin.relu_s1' % (i+1)
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

        for i in range(7):
            cov_layer_name ='net.stage1.rebnconv%d' % (i + 1)+'.relu_s1'
            print("current layer:",cov_layer_name)
            block = eval(cov_layer_name)
            cov_layer = block
            handler = cov_layer.register_forward_hook(get_feature_hook)
            u2netp_inference(net, train_loader, args.limit)
            handler.remove()

            np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer_name ='net.stage1d.rebnconv%d' % (i + 1)+'.relu_s1'
            print("current layer:",cov_layer_name)
            block = eval(cov_layer_name)
            cov_layer = block
            handler = cov_layer.register_forward_hook(get_feature_hook)
            u2netp_inference(net, train_loader, args.limit)
            handler.remove()

            np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            if i<6:
                cov_layer_name ='net.stage1.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage1d.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage2.rebnconv%d' % (i + 1)+'.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage2d.rebnconv%d' % (i + 1)+'.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

            if i<5:
                cov_layer_name ='net.stage2.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage2d.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage3.rebnconv%d' % (i + 1)+'.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage3d.rebnconv%d' % (i + 1)+'.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

            if i<4:
                cov_layer_name ='net.stage3.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage3d.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage4.rebnconv%d' % (i + 1)+'.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage4d.rebnconv%d' % (i + 1)+'.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage5.rebnconv%d' % (i + 1)+'.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage5d.rebnconv%d' % (i + 1)+'.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage6.rebnconv%d' % (i + 1)+'.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

            if i<3:
                cov_layer_name ='net.stage4.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage4d.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage5.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage5d.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer_name ='net.stage6.rebnconv%d' % (i + 1)+'d.relu_s1'
                print("current layer:",cov_layer_name)
                block = eval(cov_layer_name)
                cov_layer = block
                handler = cov_layer.register_forward_hook(get_feature_hook)
                u2netp_inference(net, train_loader, args.limit)
                handler.remove()

                np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

        cov_layer_name ='net.side1'
        print("current layer:",cov_layer_name)
        block = eval(cov_layer_name)
        cov_layer = block
        handler = cov_layer.register_forward_hook(get_feature_hook_u2net_input)
        u2netp_inference(net, train_loader, args.limit)
        handler.remove()

        np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_name ='net.side2'
        print("current layer:",cov_layer_name)
        block = eval(cov_layer_name)
        cov_layer = block
        handler = cov_layer.register_forward_hook(get_feature_hook_u2net_input)
        u2netp_inference(net, train_loader, args.limit)
        handler.remove()

        np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_name ='net.side3'
        print("current layer:",cov_layer_name)
        block = eval(cov_layer_name)
        cov_layer = block
        handler = cov_layer.register_forward_hook(get_feature_hook_u2net_input)
        u2netp_inference(net, train_loader, args.limit)
        handler.remove()

        np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_name ='net.side4'
        print("current layer:",cov_layer_name)
        block = eval(cov_layer_name)
        cov_layer = block
        handler = cov_layer.register_forward_hook(get_feature_hook_u2net_input)
        u2netp_inference(net, train_loader, args.limit)
        handler.remove()

        np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_name ='net.side5'
        print("current layer:",cov_layer_name)
        block = eval(cov_layer_name)
        cov_layer = block
        handler = cov_layer.register_forward_hook(get_feature_hook_u2net_input)
        u2netp_inference(net, train_loader, args.limit)
        handler.remove()

        np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_name ='net.side6'
        print("current layer:",cov_layer_name)
        block = eval(cov_layer_name)
        cov_layer = block
        handler = cov_layer.register_forward_hook(get_feature_hook_u2net_input)
        u2netp_inference(net, train_loader, args.limit)
        handler.remove()

        np.save('importance_score/' + args.net +'_limit%d'%(args.limit)+ '/'+cov_layer_name+'.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)


    print('The importance score generation has been completed!')