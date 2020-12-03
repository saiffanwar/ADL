from torch import nn, flatten, max as tmax
from torch.nn import functional as F
from typing import NamedTuple
import torch

class MaxOut(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        split = x.shape[1]//2
        a = x[:,:split]
        b = x[:,split:]
        return a.max(b)

class ShallowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.input_shape = ImageShape(height=height, width=width, channels=channels)
        # self.class_count = class_count
        # print(self.input_shape.channels)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(5, 5),
            padding=2,
        )
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3,3),
            padding=1,
        )
        self.initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=1,
        )
        self.initialise_layer(self.conv3)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.fc1 = nn.Linear(128*11*11, 4608)
        self.initialise_layer(self.fc1)
        self.maxout = MaxOut()
        self.fc2 = nn.Linear(2304, 2304)
        self.initialise_layer(self.fc2)


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        # print(x.size())
        # x = F.relu(x)
        # print(x.size())
        x = self.pool1(x)
        # print(x.size())

        x = self.conv2(x)
        # print(x.size())
        # x = F.relu(x)
        # print(x.size())
        x = self.pool2(x)
        # print(x.size())

        x = self.conv3(x)
        # print(x.size())
        # x = F.relu(x)
        # print(x.size())
        x = self.pool3(x)
        # print(x.size())

        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        # print(x.size())
        x = self.maxout(x)
        # x = F.relu(x)
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        x = F.sigmoid(x)
        # print(x.size())

        return x


    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
