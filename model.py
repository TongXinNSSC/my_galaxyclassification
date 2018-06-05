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
        x = self.fc(x)
        self.classifier(x)

        return x

def Resnet26(pretrained=False,num_classes=5):
    model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=5)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model