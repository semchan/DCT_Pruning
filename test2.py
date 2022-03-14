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

from data.data_loader import RescaleT
from data.data_loader import ToTensorLab
from data.data_loader import SalObjDataset
import time
cudnn.benchmark = True
cudnn.enabled=True


# def test(args):
#     net.eval()

#     cost_time = 0.
#     for i in range(20):
#         test_loss = 0
#         correct = 0
#         for data, target in val_loader:
#             if torch.cuda.is_available():
#                 data, target = data.cuda(), target.cuda()
#             if args.is_cpu:
#                 data, target = data.cpu(), target.cpu()
#                 net.cpu()
#             with torch.no_grad():
#                 data, target = Variable(data), Variable(target)
#             if args.is_cpu==False:
#                 torch.cuda.synchronize()
#                 t0 = time.time()
#                 output = net(data)
#                 torch.cuda.synchronize()
#                 cost_time += (time.time() - t0)
#             else:
#                 # torch.cuda.synchronize()
#                 t0 = time.time()
#                 output = net(data)
#                 # torch.cuda.synchronize()              
#                 cost_time += (time.time() - t0)
#             test_loss += loss_function(output, target).data.item()
#             pred = output.data.max(1)[1]
#             correct += pred.eq(target.data).cpu().sum()
#         test_loss /= len(val_loader)
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))
#     print('\n')
#     print('-------------------test result------------:')
#     print('is_org:',args.is_org)
#     print('is_cpu:',args.is_cpu)
#     print('cost time:',cost_time)
#     print('\n')
    

#     return correct.item() / len(val_loader.dataset)



def test(args):
    net.eval()

    cost_time = 0.
    count_test = 0
    while(True):
        if count_test<512*1000:
            # print('i:',i)
            if torch.cuda.is_available():
                data = torch.ones(args.batch_size, 3, 32, 32,dtype=torch.float32).cuda()
            if args.is_cpu:
                data = data.cpu()
                net.cpu()
            with torch.no_grad():
                data=Variable(data)
                if args.is_cpu==False:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    output = net(data)
                    torch.cuda.synchronize()
                    cost_time += (time.time() - t0)
                else:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    output = net(data)       
                    torch.cuda.synchronize()   
                    cost_time += (time.time() - t0)
            count_test = count_test+args.batch_size
        else:
            break
    print('\n')
    print('-------------------test result------------:')
    print('batch_size:',args.batch_size)
    print('is_org:',args.is_org)
    print('is_cpu:',args.is_cpu)
    print('cost time:',cost_time)
    print('\n')




def test_u2netp():
    net.eval()
    for i_test, data_test in tqdm(enumerate(test_salobj_dataloader)):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7


def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)

    return dn


if __name__ == '__main__':


# python test2.py --dataset 'cifar10' --data_dir './data' --batch_size 64 --test_model_dir './checkpoints/vgg_16_bn.pt' --is_cpu --compress_rate [0.]*99 --net vgg_16_bn


# python test2.py --dataset 'cifar10' --data_dir ./data --batch_size 64 --test_model_dir './pruned_models/vgg16/_6LZ3W~0/93.43.pth' --is_cpu --compress_rate [0.35]*7+[0.80]*5 --net vgg_16_bn

    parser = argparse.ArgumentParser(description='Networks Pruning')
    parser.add_argument('--dataset',type=str,default='cifar10',choices=('cifar10','imagenet','DUTS'),help='dataset')
    parser.add_argument('--data_dir',type=str,default='./data',help='path to dataset')
    parser.add_argument('--batch_size',type=int,default=512,help='batch size')
    parser.add_argument('--test_model_dir',type=str,default='./checkpoints/vgg_16_bn.pt',help='test model path')
    parser.add_argument('--is_cpu',action='store_true',help='use cpu')
    parser.add_argument('--is_org',action='store_true',help='use cpu')
    parser.add_argument('--compress_rate',type=str,default='[0.]*99',help='compress rate of each conv')
    parser.add_argument(
        '--net',
        type=str,
        default='vgg_16_bn',
        choices=('resnet_50','vgg_16_bn','resnet_56',
                'resnet_110','densenet_40','googlenet','u2netp'),
        help='net type')
    args = parser.parse_args()

    print('==> Loading data of {}..'.format(args.dataset))
    _, val_loader = load_data(args)

    print('==> Building model..')
    compress_rate = get_compress_rate(args)
    net = get_network(args, compress_rate)
    print('{}:'.format(args.net))

    if args.is_org==False:
        flops, params = profile(net, inputs=(torch.randn(1, 3, 32, 32, 
                                device='cuda' if torch.cuda.is_available() else None),))
        print('Params: %.2f' % (params))
        print('Flops: %.2f' % (flops))

    print('Compress_Rate: {}'.format(compress_rate))
    if args.dataset == 'cifar10':
        if os.path.isfile(args.test_model_dir):
            print('loading checkpoint {} ..........'.format(args.test_model_dir))
            checkpoint = torch.load(args.test_model_dir, map_location='cpu')
            if args.is_org:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net.load_state_dict(checkpoint)
                flops, params = profile(net, inputs=(torch.randn(1, 3, 32, 32, 
                                        device='cuda' if torch.cuda.is_available() else None),))
                print('Params: %.2f' % (params))
                print('Flops: %.2f' % (flops))

        else:
            print('please specify a checkpoint file')
            sys.exit()

        # flops, params = profile(net, inputs=(torch.randn(1, 3, 32, 32, 
        #                         device='cuda' if torch.cuda.is_available() else None),))
        # print('Params: %.2f' % (params))
        # print('Flops: %.2f' % (flops))

        loss_function = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss_function.cuda()

        test(args)

    if args.dataset == 'DUTS':
        prediction_dir = os.path.join(args.data_dir, args.net + '_DUTS-TE_results' + os.sep)
        image_dir = os.path.join(args.data_dir, 'DUTS-TE/DUTS-TE-Image')
        img_name_list = glob.glob(image_dir + os.sep + '*')
        test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                            lbl_name_list = [],
                                            transform=transforms.Compose([RescaleT(320),
                                                                        ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=1)

        if os.path.isfile(args.test_model_dir):
            print('loading checkpoint {} ..........'.format(args.test_model_dir))
            checkpoint = torch.load(args.test_model_dir, map_location='cpu')
            net.load_state_dict(checkpoint)
        else:
            print('please specify a checkpoint file')
            sys.exit()

        test_u2netp()

    if args.dataset == 'imagenet':

        if os.path.isfile(args.test_model_dir):
            print('loading checkpoint {} ..........'.format(args.test_model_dir))
            checkpoint = torch.load(args.test_model_dir, map_location='cpu')
            net.load_state_dict(checkpoint)
        else:
            print('please specify a checkpoint file')
            sys.exit()

        loss_function = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss_function.cuda()

        test(args)