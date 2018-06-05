import numpy as np
import matplotlib.pyplot as plt

tsne_output = np.load('tsne_output.npy')
pred_all = np.load('pred_all.npy')
#print(tsne_output)
#print(pred_all)
# tsne = pd.DataFrame(tsne.embedding_, index=data_all.index)  # 转换数据格式

colors = ['red', 'm', 'cyan', 'blue', 'lime']
colors = ['red']

plt.switch_backend('agg')
plt.figure(figsize=(10, 6))
print('start plot:')
print(tsne_output.shape)
x_min, x_max = np.min(tsne_output, 0), np.max(tsne_output, 0)
X = (tsne_output - x_min)/(x_max - x_min)
print(X.shape)
for i in range(len(colors)):
    px = []
    py = []
    px2 = []
    py2 = []

    index = np.where(pred_all[:,] == i)

    #print(pred_all[np.where(pred_all[:,] == i)].shape)
    #print(pred_all[np.where(pred_all[:,] == i)].shape)
    #print(tsne_output[np.where(pred_all[:,] == i)].shape)
    for j in range(1000):
        if pred_all[j] == i :
            #plt.plot(tsne_output[j, 0], tsne_output[j, 1])
            px.append(tsne_output[j, 0])
            py.append(tsne_output[j, 1])

    #print(px, py)
    plt.scatter(px, py, s=20, c=colors[i], marker='o')
    #plt.scatter(px2, py2, s=20, c=colors[i], marker='v')

# plt.legend(np.arange(0,5).astype(str))
plt.xticks([])
plt.yticks([])
# plt.savefig('C:/Users/Day/Desktop/PPT_report/Galaxy pic/Visualization/2/cnn1_train.png', dpi=300, bbox_inches='tight')
plt.savefig('tsne_output.png', dpi=300,
            bbox_inches='tight')
