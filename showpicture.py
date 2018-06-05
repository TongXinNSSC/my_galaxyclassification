import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

filename = 'train_list.txt'
with open(filename) as f:
    lines = f.readlines()
path,cls = lines[0].rstrip().split()
draw = mpimg.imread('./'+path)
plt.imshow(draw)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()
meta = Image.open(path)
print(meta.mode)
