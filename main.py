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
import DatasetProcessing


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

# config
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Classifier().to(device)

batch_size = 64

n_epochs = 8

patience = 5

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model. parameters(), lr=0.00025, weight_decay=1e-5)


train_set = FoodDataset("./train", tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_set = FoodDataset("./test", tfm=test_tfm)
test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset("./valid", tfm=test_tfm)
valid_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


stale = 0
best_acc = 0

for epoch in range(n_epochs):
    model.train()
