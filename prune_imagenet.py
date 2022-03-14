import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils.common import *
from utils.load_models import load_model

cudnn.benchmark = True
cudnn.enabled=True


def train(epoch):
    net.train()
    num_iter = len(train_loader)
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate: ' + str(cur_lr))
    for batch, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        adjust_learning_rate(optimizer, epoch, batch, num_iter, args)
        output = net(data)
        loss = criterion_smooth(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * len(data), len(train_loader.dataset), 100. * batch / len(train_loader), loss.data.item()))


def test():
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = net(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(val_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))

    return correct.item() / len(val_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Networks Pruning')
    parser.add_argument('--dataset',type=str,default='imagenet',choices=('cifar10','imagenet','DUTS'),help='dataset')
    parser.add_argument('--data_dir',type=str,default='./data/ImageNet',help='path to dataset')
    parser.add_argument('--job_dir',type=str,default='./save_models')
    parser.add_argument('--batch_size',type=int,default=256,help='batch size')
    parser.add_argument('--epochs',type=int,default=180,help='num of training epochs')
    parser.add_argument('--snapshot',type=int,default=20,help='save a snapshot')
    parser.add_argument('--learning_rate',type=float,default=5e-06,help='init learning rate')
    parser.add_argument('--lr_type',default='cos',type=str,help='learning rate decay schedule')
    parser.add_argument('--momentum',type=float,default=0.99,help='momentum')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay')
    parser.add_argument('--label_smooth',type=float,default=0.1,help='label smoothing')
    parser.add_argument('--resume',action='store_true',help='whether continue training from the same directory')
    parser.add_argument('--pretrain_dir',type=str,default='./checkpoints/resnet_50.pth',help='pretrain model path')
    parser.add_argument('--imp_score',type=str,default='./importance_score/resnet_50_limit5',help='importance score path')
    parser.add_argument('--compress_rate',type=str,default='[0.]+[0.1]*3+[0.4]*7+[0.4]*9',help='compress rate of each conv')
    parser.add_argument(
        '--net',
        type=str,
        default='resnet_50',
        choices=('resnet_50','vgg_16_bn','resnet_56',
                'resnet_110','densenet_40','googlenet','u2netp'),
        help='net type')
    args = parser.parse_args()
    CLASSES = 1000


    print('==> Building model..')
    compress_rate = get_compress_rate(args)
    net = get_network(args, compress_rate)
    print('{}:'.format(args.net))

    resume_epoch = 1
    load_model(args, net, resume_epoch)
    print('Compress_Rate: {}'.format(compress_rate))

    print('==> Loading data of {}..'.format(args.dataset))
    train_loader, val_loader = load_data(args)

    criterion = nn.CrossEntropyLoss()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    if torch.cuda.is_available():
        criterion.cuda()
        criterion_smooth = criterion_smooth.cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    checkpoint_path = args.job_dir

    best_acc = 0.0
    start_epoch = resume_epoch if args.resume else 1

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, args.net)):
        os.makedirs(os.path.join(checkpoint_path, args.net))
    checkpoint_path = os.path.join(checkpoint_path, args.net,'{net}-{epoch}-{acc}-regular.pth')
    best_path = args.job_dir + '/' + args.net + '/{net}-best.pth'

    print('==> Fine-tune the pruned model..')

    for epoch in range(start_epoch, args.epochs):

        train(epoch)
        acc = test()

        if best_acc < acc:
            torch.save(net.state_dict(), best_path.format(net=args.net))
            best_acc = acc

        if epoch % args.snapshot == 0 and epoch >= args.snapshot:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, acc=acc))

        print('Best Acc:{}\n'.format(best_acc))
