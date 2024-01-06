import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# custom imports
from autobrains.data_loader.augmentations import transform_resnet
from autobrains.data_loader.video_loader import VideoFrameDataset, VideoRecord
from autobrains.models.cnnlstm import CNNLSTMBaseline, CNNLSTMBaseline2
from autobrains.utils.utils import (
    convert_tensor_to_numpy,
    load_yaml,
    make_absolute_path,
)

WAYPOINTS_PATH = make_absolute_path(os.path.abspath(__file__), "../data/waypoints")
FILE_PATH = os.path.abspath(__file__)


def get_data(root_datapath: str) -> List[VideoRecord]:
    """Read the data

    Args:
        root_datapath (str): Path to the root dir of dataset

    Returns:
        List[VideoRecord]: List of Custom VideoRecord
    """
    video_instances = os.listdir(WAYPOINTS_PATH)
    video_records: List[VideoRecord] = []
    for video_instance in video_instances:
        video_records.append(
            VideoRecord(root_datapath=root_datapath, video_instance=video_instance)
        )
    return video_records


def visualize_speed(video_record: VideoRecord) -> None:
    """Visualize speed for the given scene

    Args:
        video_record (VideoRecord): Instance of VideoRecord ```e.g. 20230910-094935, 20230916-113025```
    """
    data = np.load(video_record.speed_path, allow_pickle=True)
    timestamp = np.arange(len(data))

    # plt settings
    plt.plot(timestamp, data, linestyle="-")
    plt.title("Speed v/s Time")
    plt.ylabel("Speed")
    plt.xlabel("Time")
    plt.grid(True)
    plt.show()


def visualize_waypoint(video_record: VideoRecord) -> None:
    """Visualize waypoints for the 0th and Overall trajectory

    Args:
        video_record (VideoRecord): Instance of VideoRecord ```e.g. 20230910-094935, 20230916-113025```
    """
    data = np.load(video_record.waypoints_path, allow_pickle=True)
    print(np.shape(data))

    n = 0
    plt.plot(
        np.cumsum(data[n, :, 0]),
        np.cumsum(data[n, :, 1]),
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.title(f"Trajectory_{n}")
    plt.ylabel("Longitudinal Distance")
    plt.xlabel("Lateral Distance")
    plt.grid(True)
    plt.show()

    data = np.reshape(data, (-1, 2))
    timestamp = np.arange(len(data))

    plt.plot(
        np.cumsum(data[:, 0]),
        np.cumsum(data[:, 1]),
        linestyle="-",
        color="b",
    )
    plt.title("Overall Trajectory")
    plt.ylabel("Longitudinal Distance")
    plt.xlabel("Lateral Distance")
    plt.grid(True)
    plt.show()


def visualize_rgb(root_datapath: str) -> None:
    """Visualize RGB images

    Args:
        root_datapath (str): Path to the root dir of dataset
    """

    video_frame_dataset = VideoFrameDataset(
        root_datapath=root_datapath, transform=[transform_resnet]
    )

    batch_size = 1
    data_loader = DataLoader(video_frame_dataset, batch_size=batch_size, shuffle=False)

    fig, ax = plt.subplots()
    for batch in data_loader:
        rgb: torch.tensor = batch[0]
        rgb = rgb.float().clone().detach()
        rgb_arr = rgb[0].permute(1, 2, 0).numpy()

        ax.imshow(rgb_arr)
        plt.pause(0.1)
        ax.cla()

    plt.show()


def load_model(config: dict) -> torch.nn.Module:
    """Load model with the given config

    Args:
        config (dict): config file with details

    Returns:
        torch.nn.Module: Model
    """
    model = CNNLSTMBaseline2(config["model"])
    model.load_state_dict(torch.load(config["train"]["checkpoint_path"]))
    return model


def main():
    """
    This file visualizes data:
    1. Speed
    2. RGB images
    3. Waypoints
    """

    root_datapath = "../data"
    root_datapath = os.path.join(os.path.dirname(FILE_PATH), root_datapath)

    video_records = get_data(root_datapath)

    # example to load
    # 20230910-094935 : 0
    # 20230916-113025 : 1
    example = 1

    video_record = video_records[example]
    print("video_instance: ", video_record.video_instance)

    visualize_speed(video_record)
    visualize_waypoint(video_record)
    visualize_rgb(root_datapath)


if __name__ == "__main__":
    main()
