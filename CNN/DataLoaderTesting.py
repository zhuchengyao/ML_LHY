import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm


# 定义读写文件的函数
def readfile(path, own_label):  # 后面的参数用来区分训练集和测试集
    image_dir = sorted(os.listdir(path))  # 用listdir列出path路径下的文件，然后用sorted排序。把结果赋值给image_dir
    training_array = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)  # 用np.zeros简历4维数组。其中第一维
    training_len = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        training_array[i, :, :] = cv2.resize(img, (128, 128))
        if own_label:
            training_len[i] = int(file.split("_")[0])
    if own_label:
        return training_array, training_len
    else:
        return training_array


# 分别读取训练、验证和测试的数据，保存到相应的变量里
workspace_dir = './dataset_sample'  # 数据保存的路径
print("Reading data")  # 提示读取数据
train_array, train_len = readfile(os.path.join(workspace_dir, "training"), True)  # 读取训练集数据
print("Size of training data = {}".format(len(train_array)))  # 输出训练集数据的长度
val_array, val_len = readfile(os.path.join(workspace_dir, "validation"), True)  # 读取验证集数据
print("Size of validation data = {}".format(len(val_array)))  # 输出验证集数据的长度
test_array = readfile(os.path.join(workspace_dir, "testing"), False)  # 读取测试集数据，没有label
print("Size of Testing data = {}".format(len(test_array)))  # 输出测试集数据长度

# 图像增强
train_transform = transforms.Compose([  # 定义transform为如下一系列行为：
    transforms.ToPILImage(),            # 转化成PIL图片格式
    transforms.RandomHorizontalFlip(),  # 随机将图片水平翻转
    transforms.RandomRotation(15),      # 随机旋转图片, 这段代码很坑，它会导致很多像素点数字变成0，然后你自己都不知道为什么输入发生了这样的变化。
    transforms.ToTensor(),              # 将图片转化为Tensor，并归一化
])

# 测试集图像处理
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


# 数据集类处理
class ImgDataset(Dataset):
    def __init__(self, data_array, label=None, transform=None):  # data_array就是上面准备好的图片数据的四维数列
        self.data_array = data_array
        self.label = label
        if label is not None:
            self.label = torch.LongTensor(label)  # 如果有label输入，则把label数据类型转化成tensor格式
        self.transform = transform

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, index):
        idx_array = self.data_array[index]
        if self.transform is not None:                  # 判断是否有图像变换
            idx_array = self.transform(idx_array)       # 有，则取变换后的array
        if self.label is not None:                      # 判断是否有label
            label = self.label[index]
            return idx_array, label                     # 有，则返回array 和 label
        else:
            return idx_array                            # 无label,返回array


# 初始化batch_size
batch_size = 128
# 实例化数据集
train_set = ImgDataset(train_array, train_len, train_transform)     # 用前面定义的子类加载data array， data len，
# 和图片预处理的过程
val_set = ImgDataset(val_array, val_len, test_transform)


# 载入数据集
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)   # 用DataLoader加载数据，设定好batch size和是否洗牌
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)



for data in train_loader:
    img, target = data
    print(img)
    print(target)