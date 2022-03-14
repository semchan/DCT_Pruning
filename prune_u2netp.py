import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from thop import profile
from utils.common import get_network, load_data, get_compress_rate
from utils.load_models import load_model

cudnn.benchmark = True
cudnn.enabled=True

ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000 # save the model every 2000 iterations

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


def train(epoch):
    global ite_num, ite_num4val, running_loss, running_tar_loss
    net.train()
    for i, data in enumerate(train_loader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_tar_loss += loss2.item()
        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch , args.epochs, (i + 1) * args.batch_size, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net,epoch=epoch,loss=running_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

    # scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Networks Pruning')
    parser.add_argument('--dataset',type=str,default='DUTS',choices=('cifar10','imagenet','DUTS'),help='dataset')
    parser.add_argument('--data_dir',type=str,default='./data/DUTS',help='path to dataset')
    parser.add_argument('--job_dir',type=str,default='./save_models')
    parser.add_argument('--batch_size',type=int,default=12,help='batch size')
    parser.add_argument('--epochs',type=int,default=1000,help='num of training epochs')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='init learning rate')
    # parser.add_argument('--lr_decay_step',default='50,100',type=str,help='learning rate')
    parser.add_argument('--eps',type=float,default=1e-08,help='eps')
    parser.add_argument('--weight_decay',type=float,default=0,help='weight decay')
    parser.add_argument('--resume',action='store_true',help='whether continue training from the same directory')
    parser.add_argument('--pretrain_dir',type=str,default='./checkpoints/u2netp.pth',help='pretrain model path')
    parser.add_argument('--imp_score',type=str,default='./importance_score/u2netp_limit5',help='importance score path')
    parser.add_argument('--compress_rate',type=str,default='[0.40]*40',help='compress rate of each conv')
    parser.add_argument(
        '--net',
        type=str,
        default='u2netp',
        choices=('resnet_50','vgg_16_bn','resnet_56',
                'resnet_110','densenet_40','googlenet','u2netp'),
        help='net type')
    args = parser.parse_args()


    print('==> Building model..')
    compress_rate = get_compress_rate(args)
    net = get_network(args, compress_rate)
    print('{}:'.format(args.net))

    resume_epoch = 1
    load_model(args, net, resume_epoch)
    print('Compress_Rate: {}'.format(compress_rate))

    print('==> Loading data of {}..'.format(args.dataset))
    train_loader, _ = load_data(args)
    train_num = len(train_loader) * args.batch_size

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=args.eps, weight_decay=args.weight_decay)
    # lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
    checkpoint_path = args.job_dir

    start_epoch = resume_epoch if args.resume else 1

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, args.net)):
        os.makedirs(os.path.join(checkpoint_path, args.net))
    checkpoint_path = os.path.join(checkpoint_path, args.net,'{net}-{epoch}-{loss}-regular.pth')

    print('==> Fine-tune the pruned model..')
    for epoch in range(start_epoch, args.epochs):

        train(epoch)
