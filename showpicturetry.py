# encoding: utf-8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
import  random
from utils import *
from input_data import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from MLP import Resnet26
filename = 'test_list.txt'
with open(filename) as f:
    lines = f.readlines()
test_sample = random.sample(lines,1)
path,cls = test_sample[0].rstrip().split()
# print(test_sample)
# for t in test_10:
#     with open(filename10, 'w') as file:
#         file.write(t)
# with open(filename10)
# path,cls = lines[0].rstrip().split()
# draw = mpimg.imread('./'+path)
# print('draw ',draw.size)
img = Image.open(path)
print('size :',img.size)
transform = transforms.Compose([
            transforms.CenterCrop(220),
            ResizeCV2(80, 80),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            White(),
        ])
image = transform(img)
images = image.reshape(1, 3, 64, 64)
images = torch.randn(10, 3, 64, 64)
#print('image ',images.size())
#print(images)
checkpoint = torch.load('checkpoint/_204.pth.tar' )
net = Resnet26()
# net.load_state_dict(checkpoint['state_dict'])
#own_state = net.state_dict()
#state_dict = checkpoint['state_dict']
## print(own_state.keys())
#for name, param in state_dict.items():
#    name = 'module.' + name
#    # print(name)
#    if name in own_state:
#        # print('here')
#        if isinstance(param, torch.nn.Parameter):  # isinstance函数来判断一个对象是否是一个已知的类型
#            # backwards compatibility for serialized parameters
#            param = param.data
#        try:
#            own_state[name].copy_(param)
#        except Exception:
#            print('While copying the parameter named {}, '
#                  'whose dimensions in the model are {} and '
#                  'whose dimensions in the checkpoint are {}.'
#                  .format(name, own_state[name].size(), param.size()))
#            print("But don't worry about it. Continue pretraining.")


output = net(image)
output = Variable(image)
print(output.size())
cls = int(cls)
if cls == 0:
    clsname = 'round'
elif cls == 1:
    clsname = 'middle'
elif cls == 2:
    clsname = 'cigar'
elif cls == 3:
    clsname = 'lateral'
else:
    clsname = 'spiral'

# plt.imshow(draw)  # 显示图片
# plt.axis('off')  # 不显示坐标轴
# plt.title('class: '+clsname)
# plt.show()
# meta = Image.open(path)
# print(meta.mode)
