from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import math
import random
import cv2
cv2.ocl.setUseOpenCL(False)
from torch.utils.data import Dataset
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
            return img, cls

class White(object):
    def __init__(self):
        pass
    def __call__(self, img):
        # print('img:', img)
        size = img.size()
        # print(size[0])
        img = img.view(size[0], -1)
        #print(img.size())
        eps = torch.ones(size[0],1)*(1/math.sqrt(size[1]))
        # print(eps.size())
        # print('img:', img)
        mean = torch.mean(img, dim=1, keepdim=True)
        #print('mean:', mean.size())
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
    filename = 'train_list.txt'
    train_dataset = McDataset(
        '.',
        filename,
        transforms.Compose([
            transforms.CenterCrop((170, 240)),
            ResizeCV2(80, 80),
            transforms.RandomCrop((64, 64)),
            transforms.RandomRotation(90 * random.randint(0, 4)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.8, saturation=0, hue=0),
            transforms.ToTensor(),
            White(),
        ]))

    print(train_dataset[0][0])
