import torch
import torch.nn as nn

from .base_rnn import BaseRNN, BaseRNN2, get_rnn, BaseRNN3
from .common import get_resnet, get_sequential_mlp

__all__ = ["CNNLSTMBaseline, CNNLSTMBaseline2"]


class CNNLSTMBaseline(nn.Module):
    def __init__(self, config):
        super(CNNLSTMBaseline, self).__init__()

        self.config = config

        self.resnet = get_resnet(
            self.config["resnet"]["model_name"],
            pretrained=self.config["resnet"]["pretrained"],
        )

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, self.config["input_size"])
        )

        # lstm layers
        self.lstm = BaseRNN(rnn_fn=get_rnn(config["rnn_type"]), config=config)

    def forward(self, image, speed):
        # image input
        resnet_output: torch.tensor = self.resnet(image)
        batch_size, fc_in_features = resnet_output.shape
        resnet_output = resnet_output.unsqueeze(0)

        # speed input
        speed = speed.view(1, batch_size, 1).repeat(1, 1, fc_in_features)

        # lstm input
        lstm_input = torch.cat((resnet_output, speed), dim=0)

        # forward pass
        waypoints = self.lstm(lstm_input)

        return torch.stack(waypoints, dim=1)


class CNNLSTMBaseline2(nn.Module):
    def __init__(self, config):
        super(CNNLSTMBaseline2, self).__init__()

        self.config = config

        self.resnet = get_resnet(
            self.config["resnet"]["model_name"],
            pretrained=self.config["resnet"]["pretrained"],
        )

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, self.config["input_size"])
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
        resnet_output: torch.tensor = self.resnet(image)
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

class CNNLSTMBaseline3(nn.Module):
    def __init__(self, config):
        super(CNNLSTMBaseline3, self).__init__()

        self.config = config

        self.resnet = get_resnet(
            self.config["resnet"]["model_name"],
            pretrained=self.config["resnet"]["pretrained"],
        )

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, self.config["input_size"])
        )

        self.speed_sequential_mlp = get_sequential_mlp(
            input_size=self.config["speed_sequential_mlp"]["input_size"],
            units=self.config["speed_sequential_mlp"]["units"],
            activation=self.config["speed_sequential_mlp"]["activation"],
            norm_func_name=self.config["speed_sequential_mlp"]["norm_func_name"],
            need_norm=self.config["speed_sequential_mlp"]["need_norm"],
        )
        print(self.speed_sequential_mlp)
        
        # lstm layers
        self.lstm = BaseRNN3(rnn_fn=get_rnn(config["rnn_type"]), config=config)

    def forward(self, image, speed):
        # image input
        resnet_output: torch.tensor = self.resnet(image)
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