import torch
import torch.nn as nn

from .base_rnn import BaseRNN, BaseRNN2, get_rnn
from .common import get_sequential_mlp
from .vit.vit_zoo import VitModels

class VitRNNBaseline(nn.Module):
    def __init__(self, config):
        super(VitRNNBaseline, self).__init__()

        self.config = config

        self.backbone = VitModels(
            transformer_type=self.config["vit"]["vit_model"],
            config=self.config["vit"],
            last_layer_dim=self.config["input_size"],
        )

        self.backbone.fc = nn.Sequential(
            nn.Linear(self.config["input_size"], self.config["input_size"])
        )

        self.speed_sequential_mlp = get_sequential_mlp(
            input_size=self.config["speed_sequential_mlp"]["input_size"],
            units=self.config["speed_sequential_mlp"]["units"],
            activation=self.config["speed_sequential_mlp"]["activation"],
            norm_func_name=self.config["speed_sequential_mlp"]["norm_func_name"],
            need_norm=self.config["speed_sequential_mlp"]["need_norm"],
        )

        # lstm layers
        self.lstm = BaseRNN2(rnn_fn=get_rnn(config["rnn_type"]), config=config)

    def forward(self, image, speed):
        # image input
        resnet_output: torch.tensor = self.backbone(image)
        batch_size, fc_in_features = resnet_output.shape

        # speed input
        speed = speed.view(
            batch_size, self.config["speed_sequential_mlp"]["input_size"]
        )

        speed = self.speed_sequential_mlp(speed)

        # lstm input
        lstm_input = torch.cat((resnet_output, speed), dim=1)

        # forward pass
        waypoints = self.lstm(lstm_input)

        return torch.stack(waypoints, dim=1)
