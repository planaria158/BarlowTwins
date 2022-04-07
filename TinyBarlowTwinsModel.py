import os
import numpy as np
import glob
import cv2
import sys
import torch
import torch.nn.functional as F
from torch import nn, optim

#
# Tiny Barlow Twins model
#

def build_conv_bn_block(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def build_last_conv_bn_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv0 = build_conv_bn_block(in_channels, 32)
        self.conv1 = build_conv_bn_block(32, 64)
        self.conv2 = build_conv_bn_block(64, 96)
        self.conv3 = build_conv_bn_block(96, 128)
        self.conv4 = build_conv_bn_block(128, 256)
        self.final_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x0 = self.conv0(x)
        x = F.max_pool2d(x0, kernel_size=2, stride=2)

        x1 = self.conv1(x)
        x = F.max_pool2d(x1, kernel_size=2, stride=2)

        x2 = self.conv2(x)
        x = F.max_pool2d(x2, kernel_size=2, stride=2)

        x3 = self.conv3(x)
        x = F.max_pool2d(x3, kernel_size=2, stride=2)

        x4 = self.conv4(x)
        logits = torch.squeeze(self.final_pool(x4))
        return logits 

    
    
class TinyBarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Encoder
        self.encoder = Encoder(in_channels=3) 

        # Projector
        sizes = [256] + list(map(int, args.projector.split('-')))
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
