import time
from typing import Union
import torch
from torch import nn, optim, Tensor, no_grad, save, mean, sqrt, sum as tsum
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

# load SALICON train and validation datasets
train_dataset = Salicon(
    # "train.pkl"
    "/mnt/storage/home/sa17826/ADL/cw/train.pkl"
)
test_dataset = Salicon(
    # "val.pkl"
    "/mnt/storage/home/sa17826/ADL/cw/val.pkl"
)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset_root: str,
        summary_writer: SummaryWriter,
        device: Device,
        batch_size : int = 128 
    ):
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=1,
        )
        self.val_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=True,
        )
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        log_frequency: int = 5,
        start_epoch: int = 0
        ):
        tic = time.time()
        step_start_time = time.time()
        losses = []
        accuracies = []
        lrs = np.linspace(0.03,0.0001,epochs)
        # train model for given epochs
        for epoch in range(start_epoch, epochs):
            self.model.train()
            for batch, gts in self.train_loader:
                self.optimizer = SGD(self.model.parameters(),lr=lrs[epoch], momentum=0.9, weight_decay=0.0005, nesterov=True)
                optimstate = self.optimizer.state_dict()
                self.optimizer.load_state_dict(optimstate)
                self.optimizer.zero_grad()

                batch = batch.to(self.device)
                gts = gts.to(self.device)

                output = self.model.forward(batch)
                loss = self.criterion(output,gts)
                loss.backward()
                self.optimizer.step()

                if ((self.step + 1) % log_frequency) == 0:
                    step_time = time.time() - step_start_time
                    self.log_metrics(epoch, loss, step_time)
                    self.print_metrics(epoch, loss, step_time)
                    toc = time.time() - tic
                    # print([float(loss.item()),toc, epoch])
                    losses.append([float(loss.item()),toc, epoch])
                self.step += 1

            self.summary_writer.add_scalar("epoch", epoch, self.step)

            # validate
            if ((epoch + 1) % val_frequency) == 0:
                average_loss = self.validate()
                toc = time.time() - tic
                # print([average_loss,toc])
                accuracies.append([average_loss,toc, epoch])
                self.model.train()
            # if (epoch+1) % 10 == 0:
            #     save(self.model,"checkpoint.pkl")
            #     with open('checkp_losses.pkl','wb') as file:
            #         pck.dump(losses, file)
            #     file.close()
            #     with open('checkp_accuracies.pkl','wb') as file:
            #         pck.dump(accuracies, file)
            #     file.close()

        return losses, accuracies

    def print_metrics(self, epoch, loss, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"Total training time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, loss, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "gts": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with no_grad():
            for batch, gts in self.val_loader:
                batch = batch.to(self.device)
                gts = gts.to(self.device)
                output = self.model(batch)
                loss = self.criterion(output, gts)
                total_loss += loss.item()
                preds = output.cpu().numpy()
                results["preds"].extend(list(preds))
                results["gts"].extend(list(gts.cpu().numpy()))

        average_loss = total_loss / len(self.val_loader)
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}")
        return average_loss