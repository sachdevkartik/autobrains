import logging
import math
import os
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup, get_scheduler

from autobrains.utils.common import AverageMeter, AverageMeterLoss


class BehaviourCloner:
    def __init__(
        self,
        config,
        model: nn.Module,
        train_loader,
        valid_loader,
        criterion: nn.Module,
        optimizer: nn.Module,
        use_scheduler: bool = False,
    ) -> None:
        self.model = model
        self.criterion: nn.Module = criterion
        self.optimizer: nn.Module = optimizer

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = config["device"]
        self.config = config

        self.save_checkpoint_freq: int = config["save_checkpoint_freq"]
        self.save_best_after = config["save_best_after"]
        self.checkpoint_path: str = config["checkpoint_path"]
        self.epochs: int = config["epochs"]

        self.best_loss = float("inf")
        self.epoch_loss = AverageMeter()

        self.loss_plot_path = os.path.join(self.config["plots_path"], "loss")

        self.scheduler = None
        if use_scheduler:
            self.scheduler = self.get_scheduler()

        if "load_checkpoint" in self.config.keys():
            checkpoint = torch.load(self.config["load_checkpoint"])
            self.model.load_state_dict(checkpoint)

    def train(self):
        self.epoch_losses = []
        self.val_epoch_losses = []

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = AverageMeter()
            self.model.train()

            for batch_idx, batch in enumerate(self.train_loader):
                rgb, speed, waypoints = (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                    batch[2].to(self.device),
                )

                self.optimizer.zero_grad()
                output = self.model(rgb, speed)

                loss: nn.Module = self.criterion(output, waypoints)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                epoch_loss.update(loss)

            self.epoch_losses.append(epoch_loss.avg.item())

            with torch.no_grad():
                self.model.eval()
                val_epoch_loss = AverageMeterLoss()

                for batch_idx, batch in enumerate(self.valid_loader):
                    rgb, speed, waypoints = (
                        batch[0].to(self.device),
                        batch[1].to(self.device),
                        batch[2].to(self.device),
                    )

                    output = self.model(rgb, speed)
                    val_loss = self.criterion(output, waypoints)

                    val_epoch_loss.update(val_loss)

                if val_epoch_loss.avg < self.best_loss:
                    self.best_loss = val_epoch_loss.avg
                    if epoch > self.save_best_after:
                        torch.save(self.model.state_dict(), self.checkpoint_path)
                        print("====== Model saved ======")

                self.val_epoch_losses.append(val_epoch_loss.avg.item())

            print(
                f"Epoch [{epoch}/{self.epochs}], Train Loss: {epoch_loss.avg:.7f}, Val loss : {val_epoch_loss.avg:.7f}"
            )

    def plot_loss(self):
        # epochs
        epochs = np.arange(self.epochs)

        # losses
        epoch_losses = np.array(self.epoch_losses)
        val_epoch_losses = np.array(self.val_epoch_losses)

        # plt settings
        plt.plot(epochs, epoch_losses, linestyle="-", label="Train loss")
        plt.plot(epochs, val_epoch_losses, linestyle="-", label="Val loss")

        plt.title("Losses v/s Time")
        plt.ylabel("Losses")
        plt.xlabel("Time")
        plt.legend()
        plt.grid(True)

        plt.savefig(self.loss_plot_path)
        plt.show()

    def get_scheduler(self):
        # Refer : https://www.kaggle.com/code/snnclsr/learning-rate-schedulers
        num_warmup_steps = self.config["scheduler"]["num_warmup_steps"]
        num_train_steps = math.ceil(len(self.train_loader))
        num_warmup_steps = num_train_steps * num_warmup_steps
        num_training_steps = int(num_train_steps * self.epochs)

        # learning rate scheduler
        # self.scheduler = get_cosine_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=num_warmup_steps,
        #     num_training_steps=num_training_steps,
        # )

        self.scheduler = get_scheduler(
            name=self.config["scheduler"]["name"],
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # LINEAR = "linear"
    # COSINE = "cosine"
    # COSINE_WITH_RESTARTS = "cosine_with_restarts"
    # POLYNOMIAL = "polynomial"
    # CONSTANT = "constant"
    # CONSTANT_WITH_WARMUP = "constant_with_warmup"
