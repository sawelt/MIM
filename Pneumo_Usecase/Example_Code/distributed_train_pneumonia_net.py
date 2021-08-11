import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import random
import cv2
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix

data_dir = '../chest_xray'
num_client = 4
train_dataset = ImageFolder(data_dir+'/train', transform=tt.Compose(
    [tt.Resize(255),
     tt.CenterCrop(224),
     tt.RandomHorizontalFlip(),
     tt.RandomRotation(10),
     tt.RandomGrayscale(),
     tt.RandomAffine(translate=(0.05,0.05), degrees=0),
     tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ,inplace=True),
     tt.ToTensor()
    ]))

train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
len(train_ds), len(val_ds)

batch_size=128
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
