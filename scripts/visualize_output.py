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
from autobrains.utils.utils import *

WAYPOINTS_PATH = make_absolute_path(os.path.abspath(__file__), "../data/waypoints")
FILE_PATH = os.path.abspath(__file__)


def get_data(root_datapath: str) -> List[VideoRecord]:
    video_instances = os.listdir(WAYPOINTS_PATH)
    video_records: List[VideoRecord] = []
    for video_instance in video_instances:
        video_records.append(
            VideoRecord(root_datapath=root_datapath, video_instance=video_instance)
        )
    return video_records


def load_model(config) -> torch.nn.Module:
    model = CNNLSTMBaseline2(config["model"])
    model.load_state_dict(torch.load(config["train"]["checkpoint_path"]))
    return model


def visualize_model_output(
    model: torch.nn.Module, root_datapath: str, video_record: VideoRecord, config
):
    model.to(config["device"])
    data = np.load(video_record.waypoints_path, allow_pickle=True)

    video_frame_dataset = VideoFrameDataset(
        root_datapath=root_datapath, transform=[transform_resnet]
    )

    batch_size = 1
    data_loader = DataLoader(video_frame_dataset, batch_size=batch_size, shuffle=False)

    fig, ax = plt.subplots()
    predicted_waypoints = []
    actual_waypoints = []
    print("video_record.video_instance: ", video_record.video_instance)

    model.eval()
    for batch in data_loader:
        rgb, speed, waypoints, instance = (
            batch[0].to(config["device"]),
            batch[1].to(config["device"]),
            batch[2].to(config["device"]),
            batch[3],
        )
        output = model(rgb, speed)

        if instance[0] == video_record.video_instance:
            actual_waypoints.append(convert_tensor_to_numpy(waypoints))
            predicted_waypoints.append(convert_tensor_to_numpy(output))

    # print(actual_waypoints)
    actual_waypoints_arr = np.concatenate(actual_waypoints)
    predicted_waypoints_arr = np.concatenate(predicted_waypoints)

    for n in range(20):
        plt.plot(
            actual_waypoints_arr[n, :, 0],
            actual_waypoints_arr[n, :, 1],
            linestyle="-",
            color="b",
            label="truth",
        )

        plt.plot(
            predicted_waypoints_arr[n, :, 0],
            predicted_waypoints_arr[n, :, 1],
            linestyle="-",
            color="g",
            label="prediction",
        )

        plt.title("Overall Trajectory")
        plt.ylabel("Longitudinal Distance")
        plt.xlabel("Lateral Distance")
        plt.grid(True)
        plt.legend()
        plt.show()

    predicted_waypoints_arr = np.reshape(predicted_waypoints_arr, (-1, 2))
    actual_waypoints_arr = np.reshape(actual_waypoints_arr, (-1, 2))

    plt.plot(
        np.cumsum(actual_waypoints_arr[:, 0]),
        np.cumsum(actual_waypoints_arr[:, 1]),
        linestyle="-",
        color="b",
        label="truth",
    )

    plt.plot(
        np.cumsum(predicted_waypoints_arr[:, 0]),
        np.cumsum(predicted_waypoints_arr[:, 1]),
        linestyle="-",
        color="g",
        label="prediction",
    )

    plt.title("Overall Trajectory")
    plt.ylabel("Longitudinal Distance")
    plt.xlabel("Lateral Distance")
    plt.grid(True)
    plt.legend()
    plt.show()


def main(args):
    root_datapath = "../data"
    root_datapath = os.path.join(os.path.dirname(FILE_PATH), root_datapath)

    video_records = get_data(root_datapath)
    example = 1

    video_record = video_records[example]
    print("video_instance: ", video_record.video_instance)

    # load model
    config_path = make_absolute_path(os.path.abspath(__file__), args.config)
    config = load_yaml(config_path)
    model = load_model(config)

    visualize_model_output(model, root_datapath, video_record, config)

    # visualize_rgb(root_datapath)
    # visualize_speed(video_record)
    # visualize_waypoint(video_record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../config/common.yaml",
        help="Path to the config file to visualize",
    )

    args = parser.parse_args()

    main(args)
