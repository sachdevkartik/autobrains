import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# custom imports
from autobrains.data_loader.augmentations import transform_resnet, transform_speed
from autobrains.data_loader.video_loader import VideoFrameDataset, VideoRecord
from autobrains.utils.utils import make_absolute_path

RGB_PATH = make_absolute_path(os.path.abspath(__file__), "../data/rgb")
WAYPOINTS_PATH = make_absolute_path(os.path.abspath(__file__), "../data/waypoints")
SPEED_PATH = make_absolute_path(os.path.abspath(__file__), "../data/speed")
FILE_PATH = os.path.abspath(__file__)


def test_dataset_dir():
    waypoints_contents = os.listdir(WAYPOINTS_PATH)
    speed_contents = os.listdir(SPEED_PATH)
    rgb_contents = os.listdir(RGB_PATH)

    assert (
        set(waypoints_contents) == set(speed_contents) == set(rgb_contents)
    ), "dataset dir is missing"


def test_dataset_length():
    waypoints_contents = os.listdir(WAYPOINTS_PATH)

    for content in waypoints_contents:
        waypoint_data_path = os.path.join(WAYPOINTS_PATH, content, "waypoints.npy")
        speed_data_path = os.path.join(SPEED_PATH, content, "speed.npy")
        rgb_data_path = os.path.join(RGB_PATH, content)

        # check content
        waypoint_data = np.load(waypoint_data_path, allow_pickle=True)
        speed_data = np.load(speed_data_path, allow_pickle=True)
        rgb_data_files = os.listdir(rgb_data_path)

        # assert length
        assert (
            np.shape(waypoint_data)[0] == np.shape(speed_data)[0] == len(rgb_data_files)
        ), "num of samples are inconsistent"


def test_videoRecord():
    root_datapath = "../data"
    root_datapath = os.path.join(os.path.dirname(FILE_PATH), root_datapath)

    video_instances = os.listdir(WAYPOINTS_PATH)
    video_records: List[VideoRecord] = []
    for video_instance in video_instances:
        video_records.append(
            VideoRecord(root_datapath=root_datapath, video_instance=video_instance)
        )

    waypoint_data = np.load(video_records[0].waypoints_path, allow_pickle=True)
    speed_data = np.load(video_records[0].speed_path, allow_pickle=True)

    # assert length
    assert (
        np.shape(waypoint_data)[0] == np.shape(speed_data)[0]
    ), "num of samples are inconsistent"


def test_VideoFrameDataset():
    root_datapath = "../data"
    root_datapath = os.path.join(os.path.dirname(FILE_PATH), root_datapath)
    video_frame_dataset = VideoFrameDataset(root_datapath=root_datapath)
    assert len(video_frame_dataset.video_records) == 240, "data loading not correct"


def test_VideoFrameDataset():
    root_datapath = "../data"
    root_datapath = os.path.join(os.path.dirname(FILE_PATH), root_datapath)
    video_frame_dataset = VideoFrameDataset(
        root_datapath=root_datapath, transform=[transform_resnet, transform_speed]
    )

    batch_size = 64
    data_loader = DataLoader(video_frame_dataset, batch_size=batch_size, shuffle=False)

    expected_rgb_shape = torch.Size((batch_size, 3, 224, 224))
    expected_speed_shape = torch.Size((batch_size, 1))
    expected_waypoints_shape = torch.Size((batch_size, 4, 2))

    # Iterate through batches
    for batch in data_loader:
        rgb: torch.tensor = batch[0]
        speed: torch.tensor = batch[1]
        waypoints: torch.tensor = batch[2]

        assert rgb.shape == torch.Size(
            expected_rgb_shape
        ), f"Expected RGB shape: {expected_rgb_shape}, Actual shape: {rgb.shape}"

        assert speed.shape == torch.Size(
            expected_speed_shape
        ), f"Expected Speed shape: {expected_speed_shape}, Actual shape: {speed.shape}"

        assert waypoints.shape == torch.Size(
            expected_waypoints_shape
        ), f"Expected Waypoint shape: {expected_waypoints_shape}, Actual shape: {waypoints.shape}"

        break
