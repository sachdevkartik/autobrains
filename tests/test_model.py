import os

import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from autobrains.data_loader.augmentations import transform_resnet
from autobrains.data_loader.video_loader import VideoFrameDataset, VideoRecord
from autobrains.models.cnnlstm import CNNLSTMBaseline, CNNLSTMBaseline2
from autobrains.models.vit_rnn import VitRNNBaseline
from autobrains.utils.utils import (
    combine_dict,
    load_yaml,
    make_absolute_path,
    mean_std,
    mean_std_speed,
)

FILE_PATH = os.path.abspath(__file__)


def test_CNNLSTMBaseline():
    # load configs
    baseline_config_path = make_absolute_path(
        os.path.abspath(__file__), "../config/baseline.yaml"
    )
    common_config_path = make_absolute_path(
        os.path.abspath(__file__), "../config/common.yaml"
    )

    config_model = load_yaml(baseline_config_path)
    config_common = load_yaml(common_config_path)
    config = combine_dict([config_common, config_model["model"]])

    model = CNNLSTMBaseline(config)

    batch_size = 16
    image_data = torch.randn(batch_size, 3, 224, 224)
    speed_data = torch.randn(batch_size, 1)
    expected_output_shape = torch.Size((batch_size, 4, 2))

    # forward pass
    waypoints = model(image_data, speed_data)

    assert (
        waypoints.shape == expected_output_shape
    ), f"[Shape mismatch] output shape : {waypoints.shape}; expected shape : {expected_output_shape}"


def test_VitRNNBaseline():
    # load configs
    baseline_config_path = make_absolute_path(
        os.path.abspath(__file__), "../config/levit.yaml"
    )
    common_config_path = make_absolute_path(
        os.path.abspath(__file__), "../config/common.yaml"
    )

    config_model = load_yaml(baseline_config_path)
    config_common = load_yaml(common_config_path)
    config = combine_dict([config_common, config_model["model"]])

    model = VitRNNBaseline(config)

    batch_size = 16
    image_data = torch.randn(batch_size, 3, 224, 224)
    speed_data = torch.randn(batch_size, 1)
    expected_output_shape = torch.Size((batch_size, 4, 2))

    # forward pass
    waypoints = model(image_data, speed_data)

    assert (
        waypoints.shape == expected_output_shape
    ), f"[Shape mismatch] output shape : {waypoints.shape}; expected shape : {expected_output_shape}"


def test_mean_std():
    root_datapath = "../data"
    root_datapath = os.path.join(os.path.dirname(FILE_PATH), root_datapath)
    video_frame_dataset = VideoFrameDataset(
        root_datapath=root_datapath, transform=[transform_resnet]
    )

    batch_size = len(video_frame_dataset)
    data_loader = DataLoader(video_frame_dataset, batch_size=batch_size, shuffle=False)

    mean, std = mean_std(data_loader)
    print("mean: ", mean, "std: ", std)

    mean, std = mean_std_speed(data_loader)
    print("[Speed] mean: ", mean, "std: ", std)


def test_CNNLSTMBaseline2():
    print("Testing CNNLSTMBaseline2: ")
    # load configs
    baseline_config_path = make_absolute_path(
        os.path.abspath(__file__), "../config/baseline2.yaml"
    )
    common_config_path = make_absolute_path(
        os.path.abspath(__file__), "../config/common.yaml"
    )

    config_model = load_yaml(baseline_config_path)
    config_common = load_yaml(common_config_path)
    config = combine_dict([config_common, config_model["model"]])

    model = CNNLSTMBaseline2(config)

    batch_size = 16
    image_data = torch.randn(batch_size, 3, 224, 224)
    speed_data = torch.randn(batch_size, 1)
    expected_output_shape = torch.Size((batch_size, 4, 2))

    # forward pass
    waypoints = model(image_data, speed_data)

    assert (
        waypoints.shape == expected_output_shape
    ), f"[Shape mismatch] output shape : {waypoints.shape}; expected shape : {expected_output_shape}"
