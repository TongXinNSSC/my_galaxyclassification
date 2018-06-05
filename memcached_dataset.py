import mc
from torch.utils.data import DataLoader, Dataset
import numpy as np
import io
from PIL import Image
import cv2
cv2.ocl.setUseOpenCL(False)


class ResizeCV2(object):
    def __init__(self, new_width, new_height):
        self.new_width = new_width
        self.new_height = new_height

    def __call__(self, img):
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (self.new_width, self.new_height))
        img = Image.fromarray(img_np)
        return img

def pil_loader(path):
    # buff = io.BytesIO(img_str)
    with Image.open(path) as img:
        img = img.convert('RGB')
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

    # def _init_memcached(self):
    #     if not self.initialized:
    #         server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
    #         client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
    #         self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
    #         self.initialized = True
 
    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]
        ## memcached
        try:
            # self._init_memcached()
            # value = mc.pyvector()
            # self.mclient.Get(filename, value)
            # value_str = mc.ConvertBuffer(value)
            img = pil_loader(filename)
        except:
            print("[ERROR] Image loading failed! Location: {}".format(filename))
            return self[idx - 1]
        #img = np.zeros((350, 350, 3), dtype=np.uint8)
        #img = Image.fromarray(img)
        #cls = 0
        
        ## transform
        if self.transform is not None:
            img = self.transform(img)
        if self.output_index:
            return img, cls, idx
        else:
            return img, cls
