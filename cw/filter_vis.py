import time
from typing import Union
import torch
from torch import nn, optim, Tensor, no_grad, save
from torch.nn import functional as F
from torch.optim import SGD, Adagrad
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from torch import device as Device
import numpy as np
from dataset import Salicon
import pickle as pck
from matplotlib import pyplot as plt
import os

cwd = os.getcwd()


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.savefig(cwd+'/output_images/filters.pdf')
    # plt.savefig('/mnt/storage/home/sa17826/ADL/cw/output_images/filters.pdf')


if torch.cuda.is_available():
        device = torch.device("cuda")
else:
        device = torch.device("cpu")

model1 = torch.load('model.pkl', map_location = device)
filter = model1.conv1.weight.data.clone()
visTensor(filter, ch=0, allkernels=False)

plt.axis('off')
plt.ioff()

# weight = model1.conv1.weight.data

# # normalise between 0 and 1
# min_val = torch.min(weight)
# max_val = torch.max(weight)
# norm_weight = (weight - min_val)/(max_val - min_val)

# fig, axes = plt.subplots(4,8)
# fig.suptitle("First Convolutional Layer Filters")

# for i in range(4):
#     for j in range(8):
#         filter = norm_weight[8*i+j].cpu()
#         axes[i, j].imshow(filter.permute(1,2,0))
#         axes[i, j].axis('off')
#         axes[i, j].set_title(8*i+j+1)
plt.show()
# plt.savefig(cwd+'/output_images/filters.pdf')
# print(f"Filters of first conv layer saved")
# time.sleep(10)
