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

from pathlib import Path
import argparse
import json
import math
import random
import signal
import subprocess
import time

from PIL import Image, ImageOps, ImageFilter
from BarlowTwins_CIFAR10_classifier_dataset import *
from BarlowTwins_ResNet18x1024_Classifier import *

print('torch version:', torch.__version__)
print('torchvision version:', torchvision.__version__)
print('pytorch lightning version:', pl.__version__)


# The standard values used by Albumentations Normalize method
# mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
max_pixel_value = 255.
mean = mean.unsqueeze(-1)
mean = mean.unsqueeze(-1)
std = std.unsqueeze(-1)
std = std.unsqueeze(-1)

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def main() :
    parser = argparse.ArgumentParser(description='Barlow Twins Training')
    parser.add_argument('data', type=Path, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                        help='base learning rate for weights')
    parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                        help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    # parser.add_argument('--projector', default='8192-8192-8192', type=str,
    #                     metavar='MLP', help='projector MLP')
    parser.add_argument('--projector', default='1024-1024-1024', type=str,
                        metavar='MLP', help='projector MLP')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency')
    parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')

    args = parser.parse_args(" ") 
    print(args)


    geo_transforms = A.Compose(
        [
    #      A.HorizontalFlip(p=0.5),
    #      A.VerticalFlip(p=0.5),
         A.Normalize(),
         ToTensorV2(),
        ]
    )

    pixel_transforms = A.Compose(
        [
         ToTensorV2(),
        ]
    )


    ds_train = BT_CIFAR10_Classify_Dataset(train=True, geo_transform=geo_transforms, pixel_transform=pixel_transforms)
    print('ds_train length:', ds_train.__len__())

    ds_val = BT_CIFAR10_Classify_Dataset(train=False, geo_transform=geo_transforms, pixel_transform=None)
    print('ds_val length:', ds_val.__len__())

    # Create untrained encoder-classifier model 
    model = BarlowModelClassifier_ResNet18(ds_train, ds_val, args, base_barlow_model_encoder=None)
    logger = pl.loggers.TensorBoardLogger('../lightning_logs', 'barlow_classifier')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        every_n_epochs=1,
        monitor = 'val_loss',
        mode = 'min'
    )

    trainer = pl.Trainer(max_epochs=5, strategy="dp", accelerator="gpu", devices=2, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model)

    print('Done!!')




if __name__ == '__main__':
    main()