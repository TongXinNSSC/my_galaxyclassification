#encoding:utf-8
from __future__ import print_function
#该模块提供了一种使用与操作系统相关的功能的便携方式
import os
import argparse #解析命令行参数
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
# from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import math
import random
from random import choice
from utils import *
from input_data import *
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import logging

# Training settings
# 创建一个解析对象
parser = argparse.ArgumentParser(description='galaxy')
'''添加参数
 nargs：命令行参数的个数，一般使用通配符表示，其中，'?'表示只用一个，'*'表示0到多个，'+'表示至少一个
 help： 使用这个参数描述选项作用,和ArgumentParser方法中的参数作用相似，出现的场合也一致
 dest: 这个参数相当于把位置或者选项关联到一个特定的名字
 metavar: 这个参数用于help 信息输出中
 当调用parse_args（）时，可选参数将由 - 前缀标识，其余参数将被假定为位置：
 '''
#数据集
parser.add_argument('--train-root', default='data', type=str)
#parser.add_argument('--train-source', default='data', type=str)
parser.add_argument('--val-root', default='data', type=str)
#parser.add_argument('--val-source', default='data', type=str)
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--decay_epoch', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--decay_rate', type=float, default=0.5, metavar='M',
                    help='lr decay rate (default: 0.5)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--resume', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='testing')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--multi_gpu', action='store_true', default=False,
                    help='Multi CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-path', default='checkpoint', type=str)
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--tsne', action='store_true', default=False,
                    help='tsne')

args = parser.parse_args()  #调用parse_args()方法进行解析
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
"""加载数据。组合数据集和采样器，提供数据上的单或多进程迭代器
参数：
dataset：Dataset类型，从其中加载数据
batch_size：int，可选。每个batch加载多少样本
shuffle：bool，可选。为True时表示每个epoch都对数据进行洗牌
sampler：Sampler，可选。从数据集中采样样本的方法。
num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
collate_fn：callable，可选。
pin_memory：bool，可选
drop_last：bool，可选。True表示如果最后剩下不完全的batch,丢弃。False表示不丢弃。
"""

# 数据集读取
filename = 'train_list.txt'
filename_test = 'test_list.txt'
train_dataset = McDataset(
        args.train_root,
        filename,
        transforms.Compose([
            transforms.CenterCrop(random.randint(170, 240)),
            ResizeCV2(80, 80),
            transforms.RandomCrop((64, 64)),
            transforms.RandomRotation(90*random.randint(0,4)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.8, saturation=0, hue=0),
            transforms.ToTensor(),
            White(),
        ]) )

val_dataset = McDataset(
        args.val_root,
        filename_test,
        transforms.Compose([
            transforms.CenterCrop(220),
            ResizeCV2(80, 80),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            White(),
        ]))

train_loader =  torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def pretrain(model, state_dict):
    own_state = model.state_dict()
    print(own_state.keys())
    for name, param in state_dict.items():
        #name = 'module.'+name
        #name = name[7:]
        print(name)
        if name in own_state:
            print('here')
            if isinstance(param, torch.nn.Parameter): # isinstance函数来判断一个对象是否是一个已知的类型
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(name, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")

def load_state(load_path, model, optimizer=None):
    if os.path.isfile(load_path):  #如果path是一个存在的文件，返回True。否则返回False。
        checkpoint = torch.load(load_path) # 加载整个模型
        #model.load_state_dict(checkpoint['state_dict'], strict=False)  # 仅加载模型参数
        # ckpt_keys = set(checkpoint['state_dict'].keys())
        # own_keys = set(model.state_dict().keys())
        # missing_keys = own_keys - ckpt_keys
        # for k in missing_keys:
        #     print('missing keys from checkpoint {}: {}'.format(load_path, k))
        pretrain(model, checkpoint['state_dict'])

        print("=> loaded model from checkpoint '{}'".format(load_path))  #format函数用于字符串的格式化
        if optimizer != None:
            best_prec1 = checkpoint['best_prec1']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> also loaded optimizer from checkpoint '{}' (epoch {})"
                  .format(load_path, start_epoch))
            return start_epoch
        start_epoch = 1
        return start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(load_path))
        start_epoch = 1
        return start_epoch

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels, stride=1,downsample=None, k=2):
        super(ResidualBlock,self).__init__()
        self.k = k
        # print(in_channels, self.k)
        self.in_channels = in_channels
        # print('init_in_channels:', self.in_channels)
        self.out_channels = out_channels * self.k
        self.bn1 =  nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3,padding=1)
        self.dropout = nn.Dropout(p = 0.2)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels*4, kernel_size=1,stride=stride)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        # print("block conv1 size",x.size())
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        # print("block conv2 size",x.size())
        x = self.dropout(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        # print("block conv3 size ",x.size())
        # print("bolck residual size",residual.size())
        # print("downsample ;",self.downsample)
        if self.downsample is not None:
            residual = self.downsample(residual)
        # print("x size",x.size())
        # print('residual size',residual.size())
        x += residual
        return  x

class ResNet(nn.Module):
    def __init__(self,ResidualBlock,layers,k=2, num_classes=5):
        self.k = 2
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.num_classes = num_classes
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=6, stride=1,padding=3),
            nn.MaxPool2d(kernel_size=2,stride=2),)
        self.stage2 = self._make_layer(ResidualBlock, 64, layers[0], stride=2)
        self.stage3 = self._make_layer(ResidualBlock, 128, layers[1], stride=2)
        self.stage4 = self._make_layer(ResidualBlock, 256, layers[2], stride=2)
        self.stage5 = self._make_layer(ResidualBlock, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(2048*self.k, num_classes)
        self.classifier = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

   # def _make_layer(self,block, out_channels, blocks, stride=1):
    def _make_layer(self, block, out_channels,blocks=1,stride=1):
        downsample = None
        # print('downsample: in_channel :',self.in_channels)
        # print("make layer downsample: ",downsample)
        # layers = []
        # layers.append(block(self.in_channels, out_channels, stride, downsample))
        # for i in range(1, blocks):
        #     layers.append(block(self.in_channels, out_channels))
        #     self.in_channels = out_channels
        # self.in_channels = out_channels
        # return nn.Sequential(*layers)
        layers = []
        downsample = None
        for i in range(0,blocks-1):
            if self.in_channels != out_channels * 4 * self.k:
                # print(out_channels*4*self.k)
                downsample = nn.Conv2d(self.in_channels, out_channels * 4 * self.k, kernel_size=1)
                # print('else')
            layers.append(block(self.in_channels, out_channels, stride=1, downsample=downsample))
            self.in_channels = out_channels * 4 * self.k

        downsample = None
        if stride != 1 :
            downsample = nn.MaxPool2d(kernel_size=1, stride=stride, )
        else:
            # print('else 2')
            pass
        layers.append(block(self.in_channels, out_channels, stride, downsample=downsample))
        self.in_channels = out_channels * 4 * self.k
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stage1(x)
        # print("stage1 ",x.size())
        x = self.stage2(x)
        # print("stage2 size ",x.size())
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        avgpool = x
        x = self.fc(x)
        self.classifier(x)

        return x

def Resnet26(pretrained=False,num_classes=5):
    model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=5)
    return model
def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
# class Net(nn.Module):
#     def __init__(self, num_classes=10):
#         super(Net, self).__init__()
#         self.num_classes = num_classes
#         self.stage1=nn.Sequential(
#             nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2,stride=2),)
#         self.stage2 = nn.Sequential(
#             nn.Conv2d(96,256, kernel_size=3,stride=2,padding=0),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=1,))
#         self.stage3 = nn.Sequential(
#             nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),)
#         self.stage4 = nn.Sequential(
#             nn.Conv2d(384, 384, kernel_size=2, stride=1, padding=0),
#             nn.ReLU(inplace=True),)
#         self.stage5 = nn.Linear(384,self.num_classes)
#         self.classifier = nn.Softmax(1)
#
#     #def forward(self, x):
#         x = self.stage1(x)
#         layer1_out = x
#        # print("layer1 size ",layer1_out.size())
#         x = self.stage2(x)
#         layer2_out = x
#         #print("layer2 size ", layer2_out.size())
#         x = self.stage3(x)
#         layer3_out = x
#        # print("layer3 size ", layer3_out.size())
#         x = self.stage4(x)
#         layer4_out = x
#         #print("layer4 size ", layer4_out.size())
#         x = x.view(-1, 384)
#         x = self.stage5(x)
#         layer5_out = x
#         #print("layer5 size ", layer5_out.size())
#         x = self.classifier(x)
#    #     return  x
#         return x, layer1_out, layer2_out, layer3_out, layer4_out, layer5_out
global start_epoch
start_epoch = 1

model = Resnet26()
if args.cuda:
    model.cuda()  # 将所有的模型参数移动到GPU上
    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
if args.load_path:
    if args.resume:
        start_epoch = load_state(args.load_path, model, optimizer=optimizer)
    else:
        start_epoch = load_state(args.load_path, model)
# tensorboard 可视化
tb_logger = SummaryWriter(args.save_path)
logger = create_logger('global_logger', args.save_path+'/log.txt')
logger.info('{}'.format(args))

def train(epoch):
    model.train()  # 把module设成training模式，对Dropout和BatchNorm有影响
    if epoch%args.decay_epoch == 0:
        adjust_learning_rate(optimizer, decay_rate=args.decay_rate)

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data,target = Variable(data),Variable(target)

# Variable类对Tensor对象进行封装，会保存该张量对应的梯度，以及对生成该张量的函数grad_fn的一个引用。
#         print("data.size : ",data.size())  # (64,1,28,28)
       # data = data.view(-1, 28,28)
       # print(target.size())
        optimizer.zero_grad()   # zero the gradient buffers，必须要置零
        output = model(data)
        #print('data',data.size())
        #print('output', output.size())
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        #print(output.data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        #print(pred)
        #print('predsize',pred.size())
        # output是（64，10）的tensor，pred.size是[64]
        acc = []
        recall = []
        correct = pred.eq(target.data)  #如果预测正确，correct加一
        #print(target.data.size())
        for i in range(5): #5 class
            class_tmp = torch.ones(target.data.size()[0])*i
            class_tmp = class_tmp.long()
            class_index = target.data.cpu().eq(class_tmp)
            #print(pred[pred == i].size())
            if pred[pred == i].size()[0]:
                #print(correct[class_index.cuda().byte()==1].sum())
                #print(pred[pred == i].size()[0])
                acc.append(float(correct[class_index.cuda().byte()==1].sum())/(pred[pred == i].size()[0])) # 准确率
            else:
                acc.append(0)

            #print(pred[pred == i].sum())
            #print(pred[pred == i])
            #print(target.data[target.data == i].sum())
            if target.data[target.data == i].size()[0]:
                recall.append(float(correct[class_index.cuda().byte()==1].sum())/(target.data[target.data == i].size()[0])) # 召回率
            else:
                recall.append(0)

        #print('acc', acc, 'recall', recall)
       
        
        curr_step = epoch*len(train_loader)/args.batch_size + batch_idx
        tb_logger.add_scalar('acc0_train', acc[0], curr_step)
        tb_logger.add_scalar('acc1_train', acc[1], curr_step)
        tb_logger.add_scalar('acc2_train', acc[2], curr_step)
        tb_logger.add_scalar('acc3_train', acc[3], curr_step)
        tb_logger.add_scalar('acc4_train', acc[4], curr_step)
        tb_logger.add_scalar('loss', loss.data, curr_step)
        #tb_logger.add_scalar('lr', , curr_step)
        logger.info('Loss:{loss}, curr_step:{curr_step}, accuracy:{acc}, recall:{recall}'.format(loss=loss, curr_step=curr_step, acc=acc, recall=recall))
		
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()  # 把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    test_loss = 0
    correct_num = 0
    acc, recall, pred_num, target_num = [0]*5, [0]*5, [0]*5, [0]*5

    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # volatile=true 排除子图 让test数据不参与梯度的计算,加速测试;
        # volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
       # data = data.view(-1, 28,28)
        output = model(data)
        #print('output', output)
        test_loss += F.cross_entropy(output, target).data[0]  # Variable.data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        #print('pred', pred)
        #print('predsize',pred.size())
        # output是（64，10）的tensor，pred.size是[64]
        correct_num += pred.eq(target.data).cpu().sum()  #如果预测正确，correct加一

        correct = pred.eq(target.data)  #如果预测正确，correct加一
        #print(target.data.size())
        for i in range(5): #5 class
            class_tmp = torch.ones(target.data.size()[0])*i
            class_tmp = class_tmp.long()
            class_index = target.data.cpu().eq(class_tmp)
            #print(pred[pred == i].size())
            #print(correct[class_index.cuda().byte()==1].sum())
            #print(pred[pred == i].size()[0])
            acc[i] += float(correct[class_index.cuda().byte()==1].sum())
            pred_num[i] += pred[pred == i].size()[0]

            #print(pred[pred == i].sum())
            #print(pred[pred == i])
            #print(target.data[target.data == i].sum())
            recall[i] += float(correct[class_index.cuda().byte()==1].sum())
            target_num[i] += target.data[target.data == i].size()[0]
    print(acc, recall, pred_num, target_num)
    for i in range(len(acc)):
        if pred_num[i]:
            acc[i] = acc[i]/pred_num[i]
        else:
            acc[i] = 0
        recall[i] = recall[i]/target_num[i]
    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    logger.info('\nTest set: Average loss: {:.4f}, AccuracyofAll: {}/{} ({:.0f}%), acc:{acc}, recall:{recall}\n'.format(
        test_loss, correct_num, len(test_loader.dataset),
        100. * correct_num / len(test_loader.dataset), acc=acc, recall=recall))
    print('\nTest set: Average loss: {:.4f}, AccuracyofAll: {}/{} ({:.4f}%), acc:{acc}, recall:{recall}\n'.format(
        test_loss, correct_num, len(test_loader.dataset),
        100. * correct_num / len(test_loader.dataset), acc=acc, recall=recall))

def test_with_tsne():
    #load_state(model_path, model)
    model.eval()  # 把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    test_loss = 0
    correct = 0
    data_all = Variable()
    first = True
    print('start test:')
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            #print(data.size())
        #torch.cat是把tensor联合起来，但我不知道data，和data_all有啥区别
        # print(data_all.size())

        output, avgpool = model(data)
        test_loss += F.cross_entropy(output, target).data[0]  # Variable.data
        output, avgpool = output.cpu(), avgpool.cpu()

        data, target = data.cpu(), target.cpu()
        if first :
            data_all, target_all = Variable(data), Variable(target)
            data_all = data_all.view(data_all.size()[0], -1)
            target_all = target_all.view(target_all.size()[0], -1)
        #data, target = Variable(data.cpu()), Variable(target.cpu())
        data = data.view(data.size()[0], -1)
        target = target.view(data.size()[0], -1)
        if not first:
            data_all = torch.cat((data, data_all), 0)
            target_all = torch.cat((target, target_all), 0)

        if first:
            pred_all = output.data.max(1)[1]
            tsne_input = output
            tsne_input_2 = avgpool

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        if not first:
            pred_all = torch.cat((pred, pred_all), 0)
            tsne_input = torch.cat((tsne_input, output), 0)
            tsne_input_2 = torch.cat((tsne_input_2, avgpool), 0)
            #print(tsne_input.type(), pred_all.type())
        first = False
        correct += pred.eq(target.data.cpu()).sum()
        # print(pred_all.size())
    tsne_input_3 = data_all
    # print(data_all.size())
    # print(pred_all.size())
    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print("Computing t-SNE embedding")
    # tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
    tsne_input = tsne_input.data.numpy()
    tsne_input_2 = tsne_input_2.data.numpy()
    tsne_input_3 = tsne_input_3.data.numpy()
    pred_all = pred_all.data.numpy()
    target_all = target_all.data.numpy()
    # data_all = pd.DataFrame(data_all, index=data_all[:, 0]),
    tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
    #tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    #tsne_output = np.array(tsne.fit_transform(tsne_input))
    tsne_output = np.array(tsne.fit_transform(tsne_input_2))

    print('tsne_input:', tsne_input.shape, 'tsne_input_2:', tsne_input_2.shape)
    np.save('tsne_output.npy', tsne_output)
    np.save('tsne_input.npy', tsne_input)
    np.save('tsne_input_2.npy', tsne_input_2)
    np.save('tsne_input_3.npy', tsne_input_3)
    np.save('pred_all.npy', pred_all)
    np.save('target_all.npy', target_all)
    #tsne_output = np.load('tsne_output.npy')
    print(tsne_output.shape)
    # tsne = pd.DataFrame(tsne.embedding_, index=data_all.index)  # 转换数据格式

    colors = ['red', 'm', 'cyan', 'blue', 'lime']

    plt.switch_backend('agg')
    plt.figure(figsize=(10, 6))
    print('start plot:')
    for i in range(len(colors)):
        px = []
        py = []
        px2 = []
        py2 = []

        index = np.where(pred_all[:,] == i)

        for j in range(1000):
            if pred_all[j] == i :
                #plt.plot(tsne_output[j, 0], tsne_output[j, 1])
                px.append(tsne_output[j, 0])
                py.append(tsne_output[j, 1])

        plt.scatter(px, py, s=20, c=colors[i], marker='o')
        #plt.scatter(px2, py2, s=20, c=colors[i], marker='v')

    # plt.legend(np.arange(0,5).astype(str))
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('C:/Users/Day/Desktop/PPT_report/Galaxy pic/Visualization/2/cnn1_train.png', dpi=300, bbox_inches='tight')
    plt.savefig('tsne_output.png', dpi=300,
                bbox_inches='tight')

    #plt.show()


if __name__ == '__main__' :
    # model.train()
    # input = torch.randn(10,3,64,64)
    # input = Variable(input)
    # output = model(input)
    # print(output.size())
    print('start_epoch:', start_epoch)
    if args.evaluate:
        args.epochs = 1
    for epoch in range(start_epoch, args.epochs + 1):
        if not args.evaluate:
            train(epoch)
        test(epoch)
        torch.save({
                 'epoch': epoch ,
                 'state_dict': model.state_dict(),
                 'best_prec1': 0,
                 'optimizer': optimizer.state_dict(),
             }, '%s_%s.pth.tar' % (args.save_path, epoch))
    if args.tsne:
        test_with_tsne()
