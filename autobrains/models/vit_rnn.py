import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from .base_rnn import BaseRNN, get_rnn
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

        # lstm layers
        self.lstm = BaseRNN(rnn_fn=get_rnn(config["rnn_type"]), config=config)

    def forward(self, image, speed):
        # image input
        resnet_output: torch.tensor = self.backbone(image)
        batch_size, fc_in_features = resnet_output.shape
        resnet_output = resnet_output.unsqueeze(0)

        # speed input
        speed = speed.view(1, batch_size, 1).repeat(1, 1, fc_in_features)

        # lstm input
        lstm_input = torch.cat((resnet_output, speed), dim=0)

        # forward pass
        waypoints = self.lstm(lstm_input)

        return torch.stack(waypoints, dim=1)
