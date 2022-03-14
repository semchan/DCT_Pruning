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


def train(epoch):
    net.train()
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate: ' + str(cur_lr))
    for batch, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        output = net(data)
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * len(data), len(train_loader.dataset), 100. * batch / len(train_loader), loss.data.item()))
    scheduler.step()


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
        test_loss += loss_function(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(val_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))

    return correct.item() / len(val_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Networks Pruning')
    parser.add_argument('--dataset',type=str,default='cifar10',choices=('cifar10','imagenet','DUTS'),help='dataset')
    parser.add_argument('--data_dir',type=str,default='./data',help='path to dataset')
    parser.add_argument('--job_dir',type=str,default='./save_models')
    parser.add_argument('--batch_size',type=int,default=128,help='batch size')
    parser.add_argument('--epochs',type=int,default=300,help='num of training epochs')
    parser.add_argument('--snapshot',type=int,default=20,help='save a snapshot')
    parser.add_argument('--learning_rate',type=float,default=0.01,help='init learning rate')
    parser.add_argument('--lr_decay_step',default='150,225',type=str,help='learning rate')
    parser.add_argument('--momentum',type=float,default=0.9,help='momentum')
    parser.add_argument('--weight_decay',type=float,default=0.005,help='weight decay')
    parser.add_argument('--resume',action='store_true',help='whether continue training from the same directory')
    parser.add_argument('--pretrain_dir',type=str,default='./checkpoints/googlenet.pt',help='pretrain model path')
    parser.add_argument('--imp_score',type=str,default='./importance_score/googlenet_limit5',help='importance score path')
    parser.add_argument('--compress_rate',type=str,default='[0.4]+[0.85]*2+[0.9]*5+[0.9]*2',help='compress rate of each conv')
    parser.add_argument(
        '--net',
        type=str,
        default='googlenet',
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
    flops, params = profile(net, inputs=(torch.randn(1, 3, 32, 32, 
                            device='cuda' if torch.cuda.is_available() else None),))
    print('Compress_Rate: {}'.format(compress_rate))
    print('Params: %.2f' % (params))
    print('Flops: %.2f' % (flops))

    print('==> Loading data of {}..'.format(args.dataset))
    train_loader, val_loader = load_data(args)

    loss_function = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_function.cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
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
    # for epoch in range(start_epoch):
    #     scheduler.step()

    for epoch in range(start_epoch, args.epochs):

        train(epoch)
        acc = test()

        if best_acc < acc:
            torch.save(net.state_dict(), best_path.format(net=args.net))
            best_acc = acc

        if epoch % args.snapshot == 0 and epoch >= args.snapshot:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, acc=acc))

        print('Best Acc:{}\n'.format(best_acc))
