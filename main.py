# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random


myseed = 9999   # random seed for reproducibility
torch.backends.cudnn.deterministic = True   # meaning the cnn network is fixed.
torch.backends.cudnn.benchmark = False
# if you need to reproduce the work. False means limited the algorithm to those
# of which can be reproduced.
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)  # create seed for all GPU

# data processing:

test_tfm = transforms.Compose([transforms.Resize((128, 128)),
                               transforms.ToTensor()])
# Resize function just resize the image into Hight:128, Width:128.
# ToTensor function can process PIL or ndarray to tensor.
# after the transformation, the data can be directly processed by CNN.

train_tfm = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor()])


class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x)]) for x in