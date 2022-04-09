import os
import numpy as np
import cv2
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision import transforms
import pytorch_lightning as pl


def write_image(fname, img) :
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fname, img)


#
# BarlowTwins model with ResNet50 as the encoder
#
class barlowtwins_resnet50(pl.LightningModule):
    def __init__(self, ds_train, args):
        super(barlowtwins_resnet50, self).__init__()
        self.batch_size = args.batch_size
        self.lr = 0.0001
        self.betas = [0.5, 0.999]
        self.args = args
        self.trainset = ds_train 
        self.workers = args.workers

        # The Encoder
        self.encoder = models.resnet50()
        
        # The Projector
        sizes = [1000] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
                
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
    def forward(self, y1, y2):
        z1 = self.encoder(y1)
        z1 = z1.squeeze()
        z1 = self.bn(self.projector(z1))

        z2 = self.encoder(y2)
        z2 = z2.squeeze()
        z2 = self.bn(self.projector(z2))
                
        # Empirical cross-correlation matrix
        # For example, if output of projector is z1 = [4096, 256] (batch size, embedding size)
        # then, C is z1.T * z2 :  [256, 4096] * [4096, 256] = [256, 256]
        # Then it's normalized over the batch dimension: 4096 in this case.
        # BTW: the z1, and z2 tensors are already normalized because of the last
        # layer in the encoder being a batch_norm layer. Otherwise, I'd need to 
        # normalize z1, z2 as shown in Barlow Twins paper:
        # z1_norm = (z1 - z1.mean[0])/z1.std[0]
        c = torch.mm(z1.T, z2)
        c.div_(z1.shape[0])
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()        
        loss = on_diag + self.args.lambd * off_diag
        
        return loss, on_diag, (self.args.lambd * off_diag), c
        
    def __shared_step(self, batch):
        y1, y2, _ = batch
        loss, diag, off_diag, c = self.forward(y1, y2)
        return loss, diag, off_diag, c
        
    def training_step(self, batch, batch_idx) :
        loss, diag, off_diag, c = self.__shared_step(batch)
        self.c_matrix = np.copy(c.detach().cpu().numpy())
        c_min = np.min(self.c_matrix)
        c_max = np.max(self.c_matrix)
        self.log("train_loss", loss, on_epoch=True) 
        self.log('diag', diag, on_epoch=True)
        self.log('off_diag', off_diag, on_epoch=True)
        self.log('c_max', c_max, on_epoch=True)
        self.log('c_min', c_min, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, betas=self.betas)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, 
                                      threshold=0.0001, threshold_mode='rel', cooldown=1, 
                                      min_lr=1.0e-7, eps=1e-08, verbose=True)        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}        
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, 
                          pin_memory=torch.cuda.is_available(), num_workers=self.workers)
    
#     def on_train_epoch_end(self):
#         # make the C matrix image
#         img = np.zeros((self.c_matrix.shape[0], self.c_matrix.shape[1], 3))
#         self.c_matrix = np.abs(self.c_matrix)
#         img = np.abs(self.c_matrix) * 255
#         img = img.astype('uint8')
#         fname = './lightning_logs/barlow/images_resnet18/epoch_' + str(self.current_epoch) + '.jpg'
#         write_image(fname, img)
#         return
    