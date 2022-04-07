import os
import numpy as np
import glob
import cv2
import sys
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.models as models


#
# Tiny Barlow Twins model that uses ResNet18 as the encoder
#

class BarlowTwins_ResNet18_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # The Encoder
        self.encoder = models.resnet18(zero_init_residual=True)
        self.encoder.fc = nn.Identity()
        
        # The Projector
        sizes = [512] + list(map(int, args.projector.split('-')))
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
    
    
    