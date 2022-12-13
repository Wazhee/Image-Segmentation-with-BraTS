import os
import pickle
import scipy.io
import time as time
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch.optim as optim
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn

from torchsummary import summary

"""Hyperparameters"""
DATASET_USED = '../png_dataset'
DATASET_PATH = os.path.join('dataset', DATASET_USED)

"""Loading the dataset"""
def get_indices(length, new=False):
    file_path = os.path.join('dataset', f'split_indices_{DATASET_USED}.p')

"""U-Net: Architecture"""
class UNet(nn.Module):
    """
    Imput size: (1, 512, 512)
    Output size: (2, 512, 512)
    """
    def __init__(self, nchannels, out_channels):
        super(UNet, self).__init__()
        
        padding = 1
        kernel_size = (3,3)
        max_pool = (2, 2)

        """Contracting/ Encoder"""
        # 1st layer
        self.conv1_1 = nn.Conv2d(in_channels=nchannels, out_channels=16, kernel_size=kernel_size, padding=padding)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=padding)
        self.maxpool1 = nn.MaxPool2d(max_pool)
        # 2nd layer
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding=padding)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=padding)
        self.maxpool2 = nn.MaxPool2d(max_pool)
        # 3rd layer
        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.maxpool3 = nn.MaxPool2d(max_pool)
        # 4th layer
        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=padding)
        self.maxpool4 = nn.MaxPool2d(max_pool)

        # bottle neck or 5th layer
        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=padding)
        self.conv5_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernel_size, padding=padding)
        self.upconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        """Expansive/ Decoder"""
        # 4th layer
        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=kernel_size, padding=padding)
        self.conv6_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=padding)
        self.upconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        # 3rd layer 
        self.conv7_1 = nn.Conv2d(in_channels=128, out_channels=64,kernel_size=kernel_size, padding=padding)
        self.conv7_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.upconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        # 2nd layer
        self.conv8_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernel_size, padding=padding)
        self.conv8_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=padding)
        self.upconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        # 1st layer
        self.conv9_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=kernel_size, padding=padding)
        self.conv9_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=padding)
        # output 
        self.conv10 = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

        self.relu = nn.ReLU()


    def forward(self, x):
        """Encoder"""
        # 1st Layer
        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        maxpool1 = self.maxpool1(conv1)
        # 2nd Layer
        conv2 = self.relu(self.conv2_1(maxpool1))
        conv2 = self.relu(self.conv2_2(conv2))
        maxpool2 = self.maxpool2(conv2)
        # 3rd Layer
        conv3 = self.relu(self.conv3_1(maxpool2))
        conv3 = self.relu(self.conv3_2(conv3))
        maxpool3 = self.maxpool3(conv3)
        # 4th Layer
        conv4 = self.relu(self.conv4_1(maxpool3))
        conv4 = self.relu(self.conv4_2(conv4))
        maxpool4 = self.maxpool4(conv4)

        # 5th Layer/ Bottleneck
        conv5 = self.relu(self.conv5_1(maxpool4))
        conv5 = self.relu(self.conv5_2(conv5))

        """Decoder"""
        # 4th Layer
        upconv5 = torch.cat((self.upconv1(conv5), conv4), dim=1)
        upconv5 = self.relu(self.conv6_1(upconv5))
        upconv5 = self.relu(self.conv6_2(upconv5))
        # 3rd Layer
        upconv4 = torch.cat((self.upconv2(upconv5), conv3), dim=1)
        upconv4 = self.relu(self.conv7_1(upconv4))
        upconv4 = self.relu(self.conv7_2(upconv4))
        # 2nd Layer
        upconv3 = torch.cat((self.upconv3(upconv4), conv2), dim=1)
        upconv3 = self.relu(self.conv8_1(upconv3))
        upconv3 = self.relu(self.conv8_2(upconv3))
         # 1st Layer
        upconv2 = torch.cat((self.upconv4(upconv3), conv1), dim=1)
        upconv2 = self.relu(self.conv9_1(upconv2))
        upconv2 = self.relu(self.conv9_2(upconv2))
        # ouput layer / 1st Layer
        output = self.relu(self.conv10(upconv2)) # U-Net paper solution
        # output = nn.sigmoid(self.conv10(conv9)) # Brain Tumor Segmentaiton Github Solution
        return output

    def model_summary(self, input_size=(1, 512, 512), batch_size=-1, device='cuda'):
        return summary(self, input_size, batch_size, device)