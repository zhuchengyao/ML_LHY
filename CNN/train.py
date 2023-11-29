# 导入需要用到的库
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time


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
workspace_dir = './dataset'  # 数据保存的路径
print("Reading data")  # 提示读取数据
train_array, train_len = readfile(os.path.join(workspace_dir, "training"), True)  # 读取训练集数据
print("Size of training data = {}".format(len(train_array)))  # 输出训练集数据的长度
val_array, val_len = readfile(os.path.join(workspace_dir, "validation"), True)  # 读取验证集数据
print("Size of validation data = {}".format(len(val_array)))  # 输出验证集数据的长度
test_array = readfile(os.path.join(workspace_dir, "testing"), False)  # 读取测试集数据，没有label
print("Size of Testing data = {}".format(len(test_array)))  # 输出测试集数据长度

# 图像增强
train_transform = transforms.Compose([  # 定义transform为如下一系列行为：
    transforms.ToPILImage(),  # 转化成PIL图片格式
    transforms.RandomHorizontalFlip(),  # 随机将图片水平翻转
    transforms.RandomRotation(15),  # 随机旋转图片
    transforms.ToTensor(),  # 将图片转化为Tensor，并归一化
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
        if self.transform is not None:  # 判断是否有图像变换
            idx_array = self.transform(idx_array)  # 有，则取变换后的array
        if self.label is not None:  # 判断是否有label
            label = self.label[index]
            return idx_array, label  # 有，则返回array 和 label
        else:
            return idx_array  # 无label,返回array


# 初始化batch_size
batch_size = 128
# 实例化数据集
train_set = ImgDataset(train_array, train_len, train_transform)     # 用前面定义的子类加载data array， data len，
# 和图片预处理的过程
val_set = ImgDataset(val_array, val_len, test_transform)
# 载入数据集
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)   # 用DataLoader加载数据，设定好batch size和是否洗牌
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


neural_num = 64
# 建网络模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()      # 写自己的模型的时候，一般调用父类nn.Module的构造函数
        # 建立卷积网络层
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      # 第[0]层 卷积
            nn.BatchNorm2d(64),                                                            # 第[1]层BatchNorm
            nn.ReLU(),                                                                     # ReLU激活函数
            nn.MaxPool2d(2, 2, 0),                                 # 池化

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )

        # 线性全连接网络
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),       # 原本是3通道 128*128的像素， 做了5次pooling，最终像素是4*4的。输出到1024个神经元上
            nn.ReLU(),
            nn.Linear(1024, 512),   # 全连接从1024个神经元输出到512个神经元上
            nn.ReLU(),
            nn.Linear(512, 11)      # 从512个神经元输出到11个神经元上
        )

    # 前馈，经过cnn-->view-->fc
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)       # view(a, b)函数按照
        return self.fc(out)


train_val_array = np.concatenate((train_array, val_array), axis=0)              # 合并训练集和验证集的X
train_val_len = np.concatenate((train_len, val_len), axis=0)                    # 合并训练集和验证集的Y
train_val_set = ImgDataset(train_val_array, train_val_len, train_transform)             # 实例化
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)       # 载入数据

model_best = Classifier().cuda()                                    # 此处的Classifier应该是你自己调整后，网络结构最好的网络
loss = nn.CrossEntropyLoss()                                        # 损失函数
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)     # 优化器
num_epoch = 30                                                      # 训练次数

# 开始训练
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()                                   # 定义优化器
        train_pred = model_best(data[0].cuda())                 # 计算预测值
        batch_loss = loss(train_pred, data[1].cuda())           # 预测值和真值输入loss function计算损失
        batch_loss.backward()                                   # 计算梯度
        optimizer.step()                                        # 优化权重

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    # 输出结果
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' %
          (epoch + 1, num_epoch, time.time() - epoch_start_time,
           train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))

# 测试
test_set = ImgDataset(test_array, transform=test_transform)  # 测试集实例化
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)  # 载入测试集

model_best.eval()  # 不启用 BatchNormalization 和 Dropout
prediction = []  # 新建列表，用来保存测试结果
with torch.no_grad():  # 不求导
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())  # 前馈(预测)
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)  # 即预测的label
        for y in test_label:
            prediction.append(y)

with open("predict.csv", 'w') as file:
    file.write('Id,Categroy\n')
    for i, y in enumerate(prediction):
        file.write('{},{}\n'.format(i, y))
