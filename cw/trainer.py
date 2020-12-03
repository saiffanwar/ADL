import time
from typing import Union
from torch import nn, optim, Tensor, no_grad, save
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch import device as Device
import numpy as np
from dataset import Salicon
import os
import torch
import pickle

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset_root: str,
        summary_writer: SummaryWriter,
        device: Device,
        batch_size : int = 128,
        lr : float = 0.001
    ):
            # load train/test splits of SALICON dataset
        train_dataset = Salicon(
            "/mnt/storage/home/sa17826/ADL/cw/train.pkl"
        )
        test_dataset = Salicon(
            "/mnt/storage/home/sa17826/ADL/cw/val.pkl"
        )
        
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
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr , momentum=0.9)
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        # self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            i = 0
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                logits = self.model.forward(batch)
                # print(logits, labels)
                loss = self.criterion(logits, labels)

                ## TASK 10: Compute the backward pass
                loss.backward()
                ## TASK 12: Step the optimizer and then zero out the gradient buffers.

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    # accuracy = compute_accuracy(labels, logits)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                total_loss = self.validate()
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, total_loss, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, total_loss, loss, data_load_time, step_time)


                self.step += 1
                data_load_start_time = time.time()

                self.summary_writer.add_scalar("epoch", epoch, self.step)
                if ((epoch + 1) % val_frequency) == 0:
                    self.validate() #will put the model in validation mode,
                    # so we have to switch back to train mode afterwards
                self.model.train()
                if (epoch+1)/ % 10 == 0:
                    torch.save(self.model,"checkp_model.pkl")
                i+=1
                print('Epoch '+str(epoch)+', Batch '+str(i)+' trained.')

    def print_metrics(self, epoch, total_loss, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"total_loss: {total_loss * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, total_loss, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "total_loss",
                {"train": total_loss},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

        # accuracy = compute_accuracy(
        #     np.array(results["labels"]), np.array(results["preds"])
        # )
        average_loss = total_loss / len(self.val_loader)

        # self.summary_writer.add_scalars(
        #         "accuracy",
        #         {"test": accuracy},
        #         self.step
        # )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}")


    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += float(loss.item())
                preds = labels.cpu().numpy()
                results["preds"].extend(list(preds))
                # pickle.dump(results['preds'], preds_pickle)
                results["labels"].extend(list(labels.cpu().numpy()))
        return total_loss
def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    # assert len(labels) == len(preds)
    return 0
