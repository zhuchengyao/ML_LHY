import os
import numpy as np
import cv2


def readfile(path,own_label):#后面的参数用来区分训练集和测试集
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir),128,128,3),dtype=np.uint8)
    y = np.zeros((len(image_dir)),dtype=np.uint8)
    for i,file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path,file))
        x[i,:,:] = cv2.resize(img,(128,128))
        if own_label:
            y[i] = int(file.split("_")[0])
    if own_label:
        return x,y
    else:
        return x


workspace_dir = '.\\dataset'
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
