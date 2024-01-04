import torch.nn as nn

from .common import *


def get_rnn(rnn_type: str) -> nn.Module:
    if rnn_type == "gru":
        return nn.GRU
    else:
        return nn.LSTM


class BaseRNN(nn.Module):
    def __init__(self, rnn_fn: nn.Module, config: dict) -> None:
        super(BaseRNN, self).__init__()
        self.config = config

        self.num_waypoints = self.config["num_waypoints"]
        self.dim_waypoints = self.config["dim_waypoints"]

        # lstm layers
        self.rnn_layers = nn.ModuleList(
            [
                rnn_fn(
                    input_size=self.config["input_size"],
                    hidden_size=self.config["hidden_size"],
                )
                for _ in range(self.config["num_waypoints"])
            ]
        )

        self.activation_fn = None
        if "activation_fn" in self.config:
            if self.config["activation_fn"] == "leakyrelu":
                self.activation_fn = nn.LeakyReLU()
                self.batch_norm = nn.BatchNorm2d(self.config["hidden_size"])
            elif self.config["activation_fn"] == "relu":
                self.activation_fn = nn.ReLU()
                self.batch_norm = nn.BatchNorm2d(self.config["hidden_size"])
            elif self.config["activation_fn"] == "sigmoid":
                self.activation_fn = nn.Sigmoid()
                self.batch_norm = nn.BatchNorm2d(self.config["hidden_size"])
            else:
                self.activation_fn = nn.ReLU()
                self.batch_norm = nn.BatchNorm2d(self.config["hidden_size"])

        # Fully connected layer for output
        self.fc_layers = nn.ModuleList(
            [
                nn.Linear(self.config["hidden_size"], self.config["dim_waypoints"])
                for _ in range(self.config["num_waypoints"])
            ]
        )

    def forward(self, x):
        if self.activation_fn:
            x = self.activation_fn(x)
            print("x.shape: ", x.shape)
            # x = self.batch_norm(x)

        # LSTM forward pass
        lstm_outputs = []
        for lstm_layer in self.rnn_layers:
            lstm_output = lstm_layer(x)[0]
            x = lstm_output
            lstm_outputs.append(lstm_output)

        waypoints = []
        for fc_layer, lstm_output in zip(self.fc_layers, lstm_outputs):
            waypoints.append(fc_layer(lstm_output[-1, :, :]))

        return waypoints


class BaseRNN2(nn.Module):
    def __init__(self, rnn_fn: nn.Module, config: dict) -> None:
        super(BaseRNN2, self).__init__()
        self.config = config

        self.num_waypoints = self.config["num_waypoints"]
        self.dim_waypoints = self.config["dim_waypoints"]

        self.input_size = (
            self.config["input_size"] + self.config["speed_sequential_mlp"]["units"][-1]
        )
        # start layers
        self.activation_fn = None
        if "activation_fn" in self.config:
            self.activation_fn = get_activation(self.config["activation_fn"])
            self.batch_norm = nn.BatchNorm1d(self.input_size)

        # lstm layers
        layers = []
        input_size = self.input_size
        for _ in range(self.config["num_waypoints"]):
            layers.append(
                rnn_fn(
                    input_size=input_size,
                    hidden_size=self.config["hidden_size"],
                )
            )
            input_size = self.config["hidden_size"]

        self.rnn_layers = nn.ModuleList(layers)

        self.batch_norm_rnn = nn.BatchNorm1d(self.config["hidden_size"])

        # Fully connected layer for output
        self.fc_layers = nn.ModuleList(
                [
                    nn.Linear(self.config["hidden_size"], self.config["dim_waypoints"])
                    for _ in range(self.config["num_waypoints"])
                ]
            )

    def forward(self, x):
        if self.activation_fn:
            x = self.activation_fn(x)
            x = self.batch_norm(x)

        x = x.view(-1, x.shape[0], self.input_size)

        # LSTM forward pass
        lstm_outputs = []
        for lstm_layer in self.rnn_layers:
            lstm_output = lstm_layer(x)[0]
            x = lstm_output
            lstm_outputs.append(lstm_output)

        waypoints = []
        for fc_layer, lstm_output in zip(self.fc_layers, lstm_outputs):
            waypoints.append(fc_layer(lstm_output[-1, :]))

        return waypoints
