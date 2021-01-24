#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F



torch.backends.cudnn.benchmark = True


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        self.input_shape = ImageShape(height=96, width=96, channels=3)

        #self.class_count = class_count
        ### convolution 1 and max pool ###
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2,2),
            stride=1,
        )
        self.initialise_layer(self.conv1)
        #    #self.batchNorm= nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        #### convolution 1 and max pool ###

        ### convolution 2 and max pool ###
        self.conv2= nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=64,
            kernel_size=(3,3),
            padding=(1,1),
            stride=1,
        )
        self.initialise_layer(self.conv2)
            #self.batchNorm2= nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        ### convolution 2 and max pool ###


        ### convolution 3 and max pool ###
        self.conv3= nn.Conv2d(
            in_channels= self.conv2.out_channels,
            out_channels=128,
            kernel_size=(3,3),
            stride=1,
        )
        self.initialise_layer(self.conv3)
        #        #self.batchNorm2= nn.BatchNorm2d(num_features=64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        ### convolution 3 and max pool ###


        self.fc1 = nn.Linear(12800,4608)
        self.initialise_layer(self.fc1)
        #self.batchNormFC= nn.BatchNorm1d(1024)

        ## TASK 6-1: Define the last FC layer and initialise its parameters
        self.fc2= nn.Linear(2304,2304)
        self.initialise_layer(self.fc2)



    def forward(self, images: torch.Tensor) -> torch.Tensor:

        x = self.conv1(images)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)

        x1, x2 = torch.split(x, 2304, dim=1)
        x= torch.max(x1,x2)
        #x= self.fc2(x)
        x= torch.sigmoid(self.fc2(x))
        #x= torch.nn.functional.sigmoid(torch.reshape(x, (128,1,48,48)))

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)