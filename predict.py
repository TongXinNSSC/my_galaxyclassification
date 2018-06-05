from torchvision import transforms
#from PIL import Image
import torch
import numpy as np
import math
import random
import cv2
from model import *
cv2.ocl.setUseOpenCL(False)
from torch.utils.data import Dataset
#import matplotlib.image as mpimg
# from MLP import *

class ResizeCV2(object):
    def __init__(self, new_width, new_height):
        self.new_width = new_width
        self.new_height = new_height

    def __call__(self, img):
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (self.new_width, self.new_height))
        img = Image.fromarray(img_np)
        return img

class McDataset(Dataset):
    def __init__(self, root_dir, meta_file, transform=None, output_index=False):
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        print("building dataset from %s" % meta_file)
        self.num = len(lines)
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))
        print("read meta done")
        self.initialized = False
        self.output_index = output_index

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]
        img = Image.open(filename)
        ## transform
        origin_img = (np.array(transforms.ToTensor()(img)).transpose((1,2,0)).copy()*255).astype(np.int32)
        cv2.imwrite('origin.jpg', origin_img)
        if self.transform is not None:
            img = self.transform(img)
        change_img = (np.array(img).transpose((1,2,0)).copy()*255).astype(np.int32)
        cv2.imwrite('change.jpg', change_img)
        if self.output_index:
            return img, cls, idx
        else:
            return img, cls, filename

class White(object):
    def __init__(self):
        pass
    def __call__(self, img):
        # print('img:', img)
        size = img.size()
        # print(size[0])
        img = img.view(size[0], -1)
        print(type(img))
        #print(img.size())
        eps = torch.ones(size[0],1)*(1/math.sqrt(size[1]))
        # print(eps.size())
        # print('img:', img)
        mean = torch.mean(img, dim=1, keepdim=True)
        print('mean:', mean.size())
        print(type(mean))
        std_tmp = torch.cat((torch.std(img, dim=1, keepdim=True), eps), dim=0)
        # print(torch.cat((torch.std(img, dim=0, keepdim=True), eps), dim=0).size())
        std = torch.max(std_tmp, dim=0)[0].expand_as(mean)
        # std = max(torch.std(img, dim=0, keepdim=True), eps)
        #print('std:', std.size())
        img = (img - mean) / std
        img = img.view(size[0], size[1], size[2])
        # print(img.size())
        return img

if __name__ == '__main__':
    filename = 'test_list.txt'
    test_dataset = McDataset(
        '.',
        filename,
        transforms.Compose([
            transforms.CenterCrop(220),
            ResizeCV2(80, 80),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            White(),
        ]))

    # print(train_dataset[0][0])
    num = random.randint(0, 2880)
    print('num',num)
    input = test_dataset[num][0]
    img,cls,root =test_dataset.__getitem__(num)
    print('galaxy ID : ',root[35:41])
    print('true label: ',cls)
    input = input.reshape(1,3,64,64)
    input = Variable(input)
    model = Resnet26()

    checkpoint = torch.load('checkpoint/_204.pth.tar')
    state_dict = checkpoint['state_dict']
    own_state = model.state_dict()
 #   print(own_state.keys())
    for name, param in state_dict.items():
       # name = 'module.'+name
        name = name[7:]
    #    print(name)
        if name in own_state:
       #     print('here')
            if isinstance(param, torch.nn.Parameter):  # isinstance函数来判断一个对象是否是一个已知的类型
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
#                print('here')
            except Exception:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(name, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")
    output = model(input)
    pred = output.data.max(1)[1]
    print('predict',int(pred))
    print(torch.__version__)
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
    # print(output.size())
    pred = int(pred)
    if pred == 0:
        predname = 'round'
    elif pred == 1:
        predname = 'middle'
    elif pred == 2:
        predname = 'cigar'
    elif pred == 3:
        predname = 'lateral'
    else:
        predname = 'spiral'
#     draw = mpimg.imread(root)
#     plt.imshow(draw)  # 显示图片
#     plt.axis('off')  # 不显示坐标轴
#     plt.title('true label: '+clsname+' prediction: '+ predname)
#     plt.show()

    # print(output.size())
