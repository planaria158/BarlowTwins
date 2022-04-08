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
import torchvision.models as models
from torchvision import transforms
import pytorch_lightning as pl



class resnet50_cifar10_classifier(pl.LightningModule):
    def __init__(self, ds_train, ds_val, args, base_barlow_model_encoder):
        super(resnet50_cifar10_classifier, self).__init__()
        self.batch_size = args.batch_size
        print('batch size:', self.batch_size)
        self.lr = 0.0001
        self.betas = [0.5, 0.999]
        self.args = args
        self.trainset = ds_train 
        self.valset = ds_val

        # Encoder: from previously trained BarlowTwins ResNet18 model
        if base_barlow_model_encoder == None:
            print('Creating new base encoder')
            self.encoder = models.resnet50()
            #self.encoder.fc = nn.Identity()
        else:
            print('Using the passed-in base encoder')
            self.encoder = base_barlow_model_encoder
        
        # Classification Head
        layers = []
        layers.append(nn.Linear(1000, 100, bias=False))
        layers.append(nn.BatchNorm1d(100))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(100, 10, bias=False))
        layers.append(nn.BatchNorm1d(10))
        self.classifier = nn.Sequential(*layers)
        
        
    def forward(self, x):
        return nn.functional.softmax(self.classifier(self.encoder(x)))
    
    def __shared_step(self, batch):
        imgs, y = batch
        y = (torch.nn.functional.one_hot(y, num_classes=10)).type(torch.FloatTensor).cuda()
        yhat = self.forward(imgs)
        return imgs, yhat, y
        
    def training_step(self, batch, batch_idx) :
        imgs, yhat, y = self.__shared_step(batch)
        loss = F.cross_entropy(yhat, y)
        self.log("train_loss", loss, on_epoch=True) 
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, yhat, y = self.__shared_step(batch)
        loss = F.cross_entropy(yhat, y)
        self.log("val_loss", loss, on_epoch=True) 
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, 
                                      threshold=0.0001, threshold_mode='rel', cooldown=10, 
                                      min_lr=1.0e-7, eps=1e-08, verbose=True)        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}        
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=6)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=6)       

    