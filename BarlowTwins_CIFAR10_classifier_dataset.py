import os
import numpy as np
import glob
import cv2
import sys
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


#
#  Barlow Twins CIFAR10 dataset
#  geo_transform: transform for geometric operations to image/mask pairs
#  pixel_transform: pixel-based (color etc) transform for just images.
#
class BT_CIFAR10_Classify_Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, train=True, geo_transform=None, pixel_transform=None):
        super().__init__(root='./data', train=train, download=True)
        
        self.geo_transform = geo_transform
        self.pixel_transform = pixel_transform
        self.to_tensor = transforms.Compose([transforms.ToTensor()])  
        
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        img = self.to_tensor(img)
        
        if self.geo_transform :
            transformed = self.geo_transform(image=torch.permute(img, (1,2,0)).numpy()) 
            img = transformed['image']
            
        if self.pixel_transform:
            transformed = self.pixel_transform(image=torch.permute(img, (1,2,0)).numpy()) 
            img = transformed['image']
            
        return img, label