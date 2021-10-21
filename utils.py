# 工具箱,当前包括模型保存&加载,生成数据集
import torch
import torch.utils.data as data
import os
# 保存模型
def save_models(net, type='delay'):
    PATH = './Models_of_%s' % type
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    torch.save(net.state_dict(), PATH + '/' + 'net.pt')

    print('Models saved successfully!!!')

# 加载模型
def load_models(net, type='delay'):
    PATH = './Models_of_%s' % type
    net.load_state_dict(torch.load(PATH + '/' + 'net.pt'))
    print('Models loaded successfully!!!')


class MyDataset(data.Dataset):      # 自制数据集,继承Dataset
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):   # 返回的是tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)
