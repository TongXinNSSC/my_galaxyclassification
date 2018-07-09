from torchvision import transforms
#from PIL import Image
import torch
import numpy as np
import math
import random
import cv2
import numpy as np
from loadmodel import *
from model import *
cv2.ocl.setUseOpenCL(False)
from torch.utils.data import Dataset
import logging
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
        # print(type(img))
        #print(img.size())
        eps = torch.ones(size[0],1)*(1/math.sqrt(size[1]))
        # print(eps.size())
        # print('img:', img)
        mean = torch.mean(img, dim=1, keepdim=True)
        # print('mean:', mean.size())
        # print(type(mean))
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
    filename = 'training1000_list.txt'
    cuda_flag = True 
    test_dataset = McDataset(
        '.',
        filename,
        transforms.Compose([
            transforms.CenterCrop(220),
            ResizeCV2(80, 80),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            White(),
        ]), output_index = False)
    model = Resnet26()
    model.eval()
    if cuda_flag :
        model = model.cuda()
    load_path = 'checkpoint/_204.pth.tar'
    start_epoch = load_state(load_path,model)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_flag else {}
    # 随机取一个输入求输出
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size = 1, shuffle = True, **kwargs)
    # #test(start_epoch)
    # correct_num = 0
    # for batch_idx, (data, target, fname) in enumerate(test_loader):
    #     if cuda_flag:
    #         data, target = data.cuda(), target.cuda()
    #     data, target = Variable(data), Variable(target)
    #     output = model(data)
    #     pred = output.data.max(1)[1]
    #     root = fname[0]
    #     print('galaxy ID : ', root[35:41])
    #     print('true label: ', int(target))
    #     print('predict', int(pred))
    #     # correct_num += pred.eq(target.data.view_as(pred)).long().sum()
    #     break
    # print("correct number:{} ".format(correct_num))
    #print(len(test_loader.dataset))
 #   print("correct_num: {} accracy: {:4f}".format(correct_num, 100. * float(correct_num)/len(test_loader.dataset)))


    #求training1000和test1000的输出结果
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = 1000, shuffle = False, **kwargs)
    for batch_idx, (data, target, fname) in enumerate(test_loader):
        if cuda_flag:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output, out_stage5, out_stage4, out_stage3, out_stage2, out_stage1 = model(data)
    pred = output.data.max(1)[1]
    tsne_input = output.data.numpy()
    tsne_input1 = out_stage1.data.numpy()
    tsne_input2 = out_stage2.data.numpy()
    tsne_input3 = out_stage3.data.numpy()
    tsne_input4 = out_stage4.data.numpy()
    tsne_input5 = out_stage5.data.numpy()
    tsne = manifold.TSNE(n_components=2, init='pca', learning_rate=100, random_state=0, perplexity=50,
                             early_exaggeration=1.0)
    tsne_output_origin = np.array(tsne.fit_transform(tsne_output))
    tsne_output0 = np.array(tsne.fit_transform(tsne_input))[:, np.newaxis, :]
    tsne_output1 = np.array(tsne.fit_transform(tsne_input1))[:, np.newaxis, :]
    tsne_output2 = np.array(tsne.fit_transform(tsne_input2))[:, np.newaxis, :]
    tsne_output3 = np.array(tsne.fit_transform(tsne_input3))[:, np.newaxis, :]
    tsne_output4 = np.array(tsne.fit_transform(tsne_input4))[:, np.newaxis, :]
    tsne_output5 = np.array(tsne.fit_transform(tsne_input5))[:, np.newaxis, :]
    layerout_tsne = tsne_output1
    layerout_tsne = np.concatenate((layerout_tsne, tsne_output2), axis=1)
    layerout_tsne = np.concatenate((layerout_tsne, tsne_output3), axis=1)
    layerout_tsne = np.concatenate((layerout_tsne, tsne_output4), axis=1)
    layerout_tsne = np.concatenate((layerout_tsne, tsne_output5), axis=1)
    layerout_tsne = np.concatenate((layerout_tsne, tsne_output0), axis=1)
    np.save('layerout_tsne.npy', tsne_output)
    np.save('tsne_output_origin.npy', tsne_output_origin)


    # cls = int(cls)
    # if cls == 0:
    #     clsname = 'round'
    # elif cls == 1:
    #     clsname = 'middle'
    # elif cls == 2:
    #     clsname = 'cigar'
    # elif cls == 3:
    #     clsname = 'lateral'
    # else:
    #     clsname = 'spiral'
    # print(output.size())
    # pred = int(pred)
    # if pred == 0:
    #     predname = 'round'
    # elif pred == 1:
    #     predname = 'middle'
    # elif pred == 2:
    #     predname = 'cigar'
    # elif pred == 3:
    #     predname = 'lateral'
    # else:
    #     predname = 'spiral'
#     draw = mpimg.imread(root)
#     plt.imshow(draw)  # 显示图片
#     plt.axis('off')  # 不显示坐标轴
#     plt.title('true label: '+clsname+' prediction: '+ predname)
#     plt.show()

    # print(output.size())
