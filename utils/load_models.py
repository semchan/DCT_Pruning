import torch
import torch.nn as nn
import numpy as np
import os
import copy
from collections import OrderedDict

from models.cifar10.vgg import vgg_16_bn
from models.cifar10.resnet import resnet_56, resnet_110
from models.cifar10.googlenet import googlenet, Inception
from models.cifar10.densenet import densenet_40
from models.imagenet.resnet import resnet_50
from models.DUTS.u2net import U2NETP as u2netp
from utils.common import get_network, imp_score


def load_vgg_model(model, oristate_dict, args):
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    cnt=0
    prefix = args.imp_score+'/imp_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt+=1
            oriweight = oristate_dict[name + '.weight']
            curweight =state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:

                cov_id = cnt
                print('loading imp from: ' + prefix + str(cov_id) + subfix)
                imp = np.load(prefix + str(cov_id) + subfix)
                select_index = np.argsort(imp)[orifilter_num-currentfilter_num:]  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                       state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)


def load_resnet_model(model, oristate_dict, layer, args):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.imp_score+'/imp_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt+=1
                cov_id=cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight =state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    print('loading imp from: ' + prefix + str(cov_id) + subfix)
                    imp = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(imp)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def load_google_model(model, oristate_dict, args, cpr=None):
    state_dict = model.state_dict()

    filters = [
        [64, 128, 32, 32],
        [128, 192, 96, 64],
        [192, 208, 48, 64],
        [160, 224, 64, 64],
        [128, 256, 64, 64],
        [112, 288, 64, 64],
        [256, 320, 128, 128],
        [256, 320, 128, 128],
        [384, 384, 128, 128]
    ]
    if cpr:
        for i, l in enumerate(filters):
                l[1] = int(l[1] * (1-cpr[i+1]))
                l[2] = int(l[2] * (1-cpr[i+1]))

    #last_select_index = []
    all_honey_conv_name = []
    all_honey_bn_name = []
    cur_last_select_index = []

    cnt=0
    prefix = args.imp_score+'/imp_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, Inception):

            cnt += 1
            cov_id = cnt

            honey_filter_channel_index = [
                '.branch5x5.6',
            ]  # the index of sketch filter and channel weight
            honey_channel_index = [
                '.branch1x1.0',
                '.branch3x3.0',
                '.branch5x5.0',
                '.branch_pool.1'
            ]  # the index of sketch channel weight
            honey_filter_index = [
                '.branch3x3.3',
                '.branch5x5.3',
            ]  # the index of sketch filter weight
            honey_bn_index = [
                '.branch3x3.4',
                '.branch5x5.4',
                '.branch5x5.7',
            ]  # the index of sketch bn weight

            for bn_index in honey_bn_index:
                all_honey_bn_name.append(name + bn_index)

            last_select_index = cur_last_select_index[:]
            cur_last_select_index=[]

            for weight_index in honey_channel_index:

                if '3x3' in weight_index:
                    branch_name='_n3x3'
                elif '5x5' in weight_index:
                    branch_name='_n5x5'
                elif '1x1' in weight_index:
                    branch_name='_n1x1'
                elif 'pool' in weight_index:
                    branch_name='_pool_planes'

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight =state_dict[conv_name]
                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num:
                    select_index = last_select_index
                else:
                    select_index = list(range(0, orifilter_num))

                for i in range(state_dict[conv_name].size(0)):
                    for index_j, j in enumerate(select_index):
                        state_dict[conv_name][i][index_j] = \
                            oristate_dict[conv_name][i][j]

                if branch_name=='_n1x1':
                    tmp_select_index = list(range(state_dict[conv_name].size(0)))
                    cur_last_select_index += tmp_select_index
                if branch_name=='_pool_planes':
                    tmp_select_index = list(range(state_dict[conv_name].size(0)))
                    tmp_select_index = [x+filters[cov_id-2][0]+filters[cov_id-2][1]+filters[cov_id-2][2] for x in tmp_select_index]
                    cur_last_select_index += tmp_select_index

            for weight_index in honey_filter_index:

                if '3x3' in weight_index:
                    branch_name='_n3x3'
                elif '5x5' in weight_index:
                    branch_name='_n5x5'
                elif '1x1' in weight_index:
                    branch_name='_n1x1'
                elif 'pool' in weight_index:
                    branch_name='_pool_planes'

                conv_name = name + weight_index + '.weight'

                all_honey_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight =state_dict[conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    print('loading imp from: ' + prefix + str(cov_id) + branch_name + subfix)
                    imp = np.load(prefix + str(cov_id)  + branch_name + subfix)
                    select_index = np.argsort(imp)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                else:
                    select_index = list(range(0, orifilter_num))

                for index_i, i in enumerate(select_index):
                    state_dict[conv_name][index_i] = \
                        oristate_dict[conv_name][i]

                if branch_name=='_n3x3':
                    tmp_select_index = [x+filters[cov_id-2][0] for x in select_index]
                    cur_last_select_index += tmp_select_index
                if branch_name=='_n5x5':
                    last_select_index=select_index

            for weight_index in honey_filter_channel_index:

                if '3x3' in weight_index:
                    branch_name='_n3x3'
                elif '5x5' in weight_index:
                    branch_name='_n5x5'
                elif '1x1' in weight_index:
                    branch_name='_n1x1'
                elif 'pool' in weight_index:
                    branch_name='_pool_planes'

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num:
                    select_index = last_select_index
                else:
                    select_index = range(0, orifilter_num)

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                select_index_1 = copy.deepcopy(select_index)

                if orifilter_num != currentfilter_num:
                    print('loading imp from: ' + prefix + str(cov_id) + branch_name + subfix)
                    imp = np.load(prefix + str(cov_id) + branch_name + subfix)
                    select_index = np.argsort(imp)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                else:
                    select_index = list(range(0, orifilter_num))

                if branch_name == '_n5x5':
                    tmp_select_index = [x+filters[cov_id-2][0]+filters[cov_id-2][1] for x in select_index]
                    cur_last_select_index += tmp_select_index

                for index_i, i in enumerate(select_index):
                    for index_j, j in enumerate(select_index_1):
                        state_dict[conv_name][index_i][index_j] = \
                            oristate_dict[conv_name][i][j]

        elif name=='pre_layers':

            cnt += 1
            cov_id = cnt

            honey_filter_index = ['.0']  # the index of sketch filter weight
            honey_bn_index = ['.1']  # the index of sketch bn weight

            for bn_index in honey_bn_index:
                all_honey_bn_name.append(name + bn_index)

            for weight_index in honey_filter_index:

                conv_name = name + weight_index + '.weight'

                all_honey_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight =state_dict[conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    imp = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(imp)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    cur_last_select_index = select_index[:]

                    for index_i, i in enumerate(select_index):
                       state_dict[conv_name][index_i] = \
                            oristate_dict[conv_name][i]#'''

    for name, module in model.named_modules():  # Reassign non sketch weights to the new network
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            if name not in all_honey_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):

            if name not in all_honey_bn_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name + '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def load_densenet_model(model, oristate_dict, args):

    state_dict = model.state_dict()
    last_select_index = [] #Conv index selected in the previous layer

    cnt=0
    prefix = args.imp_score+'/imp_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt+=1
            cov_id = cnt
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:
                print('loading imp from: ' + prefix + str(cov_id) + subfix)
                imp = np.load(prefix + str(cov_id) + subfix)
                select_index = list(np.argsort(imp)[orifilter_num-currentfilter_num:])  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
                select_index = list(range(0, orifilter_num))

            else:
                select_index = list(range(0, orifilter_num))
                state_dict[name + '.weight'] = oriweight

            if cov_id==1 or cov_id==14 or cov_id==27:
                last_select_index = select_index
            else:
                tmp_select_index = [x+cov_id*12-(cov_id-1)//13*12 for x in select_index]
                last_select_index += tmp_select_index

    model.load_state_dict(state_dict)


def load_resnet_imagenet_model(model, oristate_dict, args):
    cfg = {'resnet18': [2, 2, 2, 2],
           'resnet34': [3, 4, 6, 3],
           'resnet_50': [3, 4, 6, 3],
           'resnet101': [3, 4, 23, 3],
           'resnet152': [3, 8, 36, 3]}

    state_dict = model.state_dict()

    current_cfg = cfg[args.net]
    last_select_index = None

    all_honey_conv_weight = []

    bn_part_name=['.weight','.bias','.running_mean','.running_var']
    prefix = args.imp_score+'/imp_conv'
    subfix = ".npy"
    cnt=1

    conv_weight_name = 'conv1.weight'
    all_honey_conv_weight.append(conv_weight_name)
    oriweight = oristate_dict[conv_weight_name]
    curweight = state_dict[conv_weight_name]
    orifilter_num = oriweight.size(0)
    currentfilter_num = curweight.size(0)

    if orifilter_num != currentfilter_num:
        print('loading imp from: ' + prefix + str(cnt) + subfix)
        imp = np.load(prefix + str(cnt) + subfix)
        select_index = np.argsort(imp)[orifilter_num - currentfilter_num:]  # preserved filter id
        select_index.sort()

        for index_i, i in enumerate(select_index):
            state_dict[conv_weight_name][index_i] = \
                oristate_dict[conv_weight_name][i]
            for bn_part in bn_part_name:
                state_dict['bn1' + bn_part][index_i] = \
                    oristate_dict['bn1' + bn_part][i]

        last_select_index = select_index
    else:
        state_dict[conv_weight_name] = oriweight
        for bn_part in bn_part_name:
            state_dict['bn1' + bn_part] = oristate_dict['bn1'+bn_part]

    state_dict['bn1' + '.num_batches_tracked'] = oristate_dict['bn1' + '.num_batches_tracked']

    cnt+=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'

        for k in range(num):
            if args.net == 'resnet_18' or args.net == 'resnet_34':
                iter = 2  # the number of convolution layers in a block, except for shortcut
            else:
                iter = 3
            if k==0:
                iter +=1
            for l in range(iter):
                record_last=True
                if k==0 and l==2:
                    conv_name = layer_name + str(k) + '.downsample.0'
                    bn_name = layer_name + str(k) + '.downsample.1'
                    record_last=False
                elif k==0 and l==3:
                    conv_name = layer_name + str(k) + '.conv' + str(l)
                    bn_name = layer_name + str(k) + '.bn' + str(l)
                else:
                    conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                    bn_name = layer_name + str(k) + '.bn' + str(l + 1)

                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    print('loading imp from: ' + prefix + str(cnt) + subfix)
                    imp = np.load(prefix + str(cnt) + subfix)
                    select_index = np.argsort(imp)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]

                            for bn_part in bn_part_name:
                                state_dict[bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]

                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                            for bn_part in bn_part_name:
                                state_dict[bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]

                    if record_last:
                        last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]

                    for bn_part in bn_part_name:
                        state_dict[bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]

                    if record_last:
                        last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    for bn_part in bn_part_name:
                        state_dict[bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]
                    if record_last:
                        last_select_index = None

                state_dict[bn_name + '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']
                cnt+=1

    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def load_u2netp_model(model, oristate_dict, args):
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    cnt = 0
    stage_id = 1
    save_select_index = []
    save_stage_select_index = []
    save_side_select_index = []

    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if name == 'outconv':
            break

        if isinstance(module, nn.Conv2d):
            side_name = name.split('.')[0]
            decode = True if side_name[-1] == 'd' else False
            flag = '.' if not decode else 'd.'

            prefix = args.imp_score+'/net.stage'
            midfix = name.split('.')[1] if not side_name[:4] == 'side' else None
            subfix = ".relu_s1.npy"
            oriweight = oristate_dict[name + '.weight']
            curweight =state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)
            oriin_num = oriweight.size(1)

            cov_id = midfix[-2:] if not side_name[:4] == 'side' else None
            if decode == True and side_name[-2] != str(stage_id):
                stage_id -= 1
                save_side_select_index.append(list(last_select_index))
                save_select_index = []
            elif decode == False and side_name[-1] != str(stage_id):
                stage_id += 1
                save_stage_select_index.append(list(last_select_index))
                save_select_index = []

            if midfix != None:

                if decode == True and cov_id == 'in':
                    if orifilter_num != currentfilter_num:
                        print('loading rank from: ' + prefix + str(stage_id) + flag + midfix + subfix)
                        rank = np.load(prefix + str(stage_id) + flag + midfix + subfix)
                        select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                        select_index.sort()

                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name + '.weight'][index_i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                            for index_k, k in enumerate(save_stage_select_index[stage_id-1]):
                                state_dict[name + '.weight'][index_i][index_k+len(last_select_index)] = \
                                    oristate_dict[name + '.weight'][i][k+int(oriin_num/2)]

                        last_select_index = select_index
                        save_select_index.append(list(select_index))

                    elif last_select_index is not None:
                        print('loading rank from: ' + prefix + str(stage_id) + flag + midfix + subfix)
                        rank = np.load(prefix + str(stage_id) + flag + midfix + subfix)
                        select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                        select_index.sort()

                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                            for index_k, k in enumerate(save_select_index[int(cov_id[0])]):
                                state_dict[name + '.weight'][i][index_k+len(last_select_index)] = \
                                    oristate_dict[name + '.weight'][i][k+int(oriin_num/2)]

                        last_select_index = select_index
                        save_select_index.append(list(select_index))

                    else:
                        state_dict[name + '.weight'] = oriweight
                        last_select_index = None
                        save_select_index.append(None)

                elif cov_id[1] != 'd':
                    if orifilter_num != currentfilter_num:
                        print('loading rank from: ' + prefix + str(stage_id) + flag + midfix + subfix)
                        rank = np.load(prefix + str(stage_id) + flag + midfix + subfix)
                        select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                        select_index.sort()

                        if last_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_select_index):
                                    state_dict[name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_select_index = select_index
                        save_select_index.append(list(select_index))

                    elif last_select_index is not None:
                        print('loading rank from: ' + prefix + str(stage_id) + flag + midfix + subfix)
                        rank = np.load(prefix + str(stage_id) + flag + midfix + subfix)
                        select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                        select_index.sort()

                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]

                        last_select_index = select_index
                        save_select_index.append(list(select_index))

                    else:
                        state_dict[name + '.weight'] = oriweight
                        last_select_index = None
                        save_select_index.append(None)

                else:
                    if orifilter_num != currentfilter_num:
                        print('loading rank from: ' + prefix + str(stage_id) + flag + midfix + subfix)
                        rank = np.load(prefix + str(stage_id) + flag + midfix + subfix)
                        select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                        select_index.sort()

                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name + '.weight'][index_i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                            for index_k, k in enumerate(save_select_index[int(cov_id[0])]):
                                state_dict[name + '.weight'][index_i][index_k+len(last_select_index)] = \
                                    oristate_dict[name + '.weight'][i][k+int(oriin_num/2)]

                        last_select_index = select_index

                    elif last_select_index is not None:
                        print('loading rank from: ' + prefix + str(stage_id) + flag + midfix + subfix)
                        rank = np.load(prefix + str(stage_id) + flag + midfix + subfix)
                        select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                        select_index.sort()

                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                            for index_k, k in enumerate(save_select_index[int(cov_id[0])]):
                                state_dict[name + '.weight'][i][index_k+len(last_select_index)] = \
                                    oristate_dict[name + '.weight'][i][k+int(oriin_num/2)]

                        last_select_index = select_index

                    else:
                        state_dict[name + '.weight'] = oriweight
                        last_select_index = None

            else:
                cnt += 1
                if orifilter_num != currentfilter_num:
                    print('loading rank from: ' + prefix[:-5] + 'side%d.npy'%(cnt))
                    rank = np.load(prefix[:-5] + 'side%d.npy'%(cnt))
                    select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name + '.weight'][index_i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name + '.weight'][index_i] = \
                                oristate_dict[name + '.weight'][i]

                elif last_select_index is not None:
                    for i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]

                else:
                    state_dict[name + '.weight'] = oriweight

                last_select_index = save_side_select_index[5-cnt]


    model.load_state_dict(state_dict)


def load_model(args, model, resume_epoch=1):
    
    if args.resume:
        epoch_max = 0
        ckp_list = os.listdir(os.path.join(args.job_dir, args.net))
        for ckp in ckp_list:
            if ckp.split('-')[0] == args.net:
                epoch_num = int(ckp.split('-')[1])
                if epoch_num > epoch_max:
                    epoch_max = epoch_num
                    ckp_max = ckp

        checkpoint_dir = os.path.join(args.job_dir, ckp_max)
        resume_epoch = epoch_max

        print('loading checkpoint {} ..........'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)

        # deal with the single-multi GPU problem
        new_state_dict = OrderedDict()
        tmp_ckpt = checkpoint['state_dict']

        for k, v in tmp_ckpt.items():
            new_state_dict[k.replace('module.', '')] = v

        model.load_state_dict(new_state_dict)
        print("loaded checkpoint {} epoch = {}".format(checkpoint_dir, epoch_max))

    else:
        if os.path.exists(args.pretrain_dir):
            print('resuming from pretrain model')
            origin_model = eval(args.net)(compress_rate=[0.] * 100).cuda()
            ckpt = torch.load(args.pretrain_dir, map_location='cuda:0')

            if args.net == 'densenet_40' or args.net == 'resnet_110':
                new_state_dict = OrderedDict()
                for k, v in ckpt['state_dict'].items():
                    new_state_dict[k.replace('module.', '')] = v
                origin_model.load_state_dict(new_state_dict)
            elif args.net=='resnet_50' or args.net=='u2netp':
                origin_model.load_state_dict(ckpt)
            else:
                origin_model.load_state_dict(ckpt['state_dict'])

            imp_score(origin_model, args)
            oristate_dict = origin_model.state_dict()

            if args.net == 'googlenet':
                load_google_model(model, oristate_dict, args)
            elif args.net == 'vgg_16_bn':
                load_vgg_model(model, oristate_dict, args)
            elif args.net == 'resnet_56':
                load_resnet_model(model, oristate_dict, 56, args)
            elif args.net == 'resnet_110':
                load_resnet_model(model, oristate_dict, 110, args)
            elif args.net == 'densenet_40':
                load_densenet_model(model, oristate_dict, args)
            elif args.net == 'resnet_50':
                load_resnet_imagenet_model(model, oristate_dict, args)
            elif args.net == 'u2netp':
                load_u2netp_model(model, oristate_dict, args)
            else:
                raise NotImplementedError
        else:
            print('please check the path of pretrain_model')
            raise NotImplementedError