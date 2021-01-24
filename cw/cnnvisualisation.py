import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from dataset import Salicon
from torch.utils.data import DataLoader


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(5, 5),
            padding=2,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # self.batch2d1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels= 64,
            kernel_size = (3,3),
            padding = 1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        # self.batch2d2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels= 128,
            kernel_size = (3,3),
            padding = 1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        # self.batch2d3 = nn.BatchNorm2d(num_features=128)

        self.fc1 = nn.Linear(128*11*11, 48*48*2)
        self.batch1d = nn.BatchNorm1d(num_features=4608)
        self.fc2 = nn.Linear(48*48, 48*48)
        self.initialise_layer(self.fc1)


    def forward(self, x):
        x = self.conv1(x)
        # print(x.size())
        x = F.relu(x)
        # print(x.size())
        x = self.pool1(x)
        # print(x.size())
        # x = self.batch2d1(x)
        x = self.conv2(x)
        # print(x.size())
        x = F.relu(x)
        # print(x.size())
        x = self.pool2(x)
        # print(x.size())
        # x = self.batch2d2(x)
        x = self.conv3(x)
        # print(x.size())
        x = F.relu(x)
        # print(x.size())
        x = self.pool3(x)
        # print(x.size())
        # x = self.batch2d3(x)

        x = torch.flatten(x,start_dim=1)
        # print(x.size())
        x = self.fc1(x)
        # print(x.size())
        # x = self.batch1d(x)
        x1, x2 = torch.split(x, 2304, dim=1)
        x = torch.max(x1,x2)
        x = F.relu(x)
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        # print(x.size())

        return x
        
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
# dataset = datasets.MNIST(
#     root='PATH',
#     transform=transforms.ToTensor()
# )
# loader = DataLoader(
#     dataset,
#     num_workers=2,
#     batch_size=8,
#     shuffle=True
# )
# if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

train_dataset = Salicon(
"train.pkl"
# "/mnt/storage/home/sa17826/ADL/cw/train.pkl"
)
        
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=128,
    pin_memory=True,
    num_workers=1,
)
model = MyModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)

epochs = 1
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        
        print('Epoch {}, Batch idx {}, loss {}'.format(
            epoch, batch_idx, loss.item()))


def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img

# Plot some images
idx = torch.randint(0, output.size(0), ())
pred = normalize_output(output[idx, 0])
img = data[idx, 0]

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(img.detach().numpy())
axarr[1].imshow(pred.detach().numpy())

# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))
data, _ = dataset[0]
data.unsqueeze_(0)
output = model(data)

act = activation['conv1'].squeeze()
fig, axarr = plt.subplots(act.size(0))
for idx in range(act.size(0)):
    axarr[idx].imshow(act[idx])