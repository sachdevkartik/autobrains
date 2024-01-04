import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, random_split

from autobrains.data_loader.data_classes import FrameData, VideoRecord


def make_absolute_path(root_path: str, rel_path: str) -> str:
    return os.path.join(os.path.dirname(root_path), rel_path)


def make_video_record_list(data: VideoRecord) -> List[FrameData]:
    frame_data_list: List[FrameData] = []

    waypoints_data = np.load(data.waypoints_path, allow_pickle=True)
    speed_data = np.load(data.speed_path, allow_pickle=True)

    for i in range(len(data.rgb_path_list)):
        frame_data = FrameData()
        frame_data.rgb_path = data.rgb_path_list[i]
        frame_data.speed = speed_data[i]
        frame_data.waypoints = waypoints_data[i]
        frame_data.instance = data.video_instance
        frame_data.frame_idx = i

        frame_data_list.append(frame_data)

    return frame_data_list


def load_yaml(path: str) -> dict:
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data


def load_yaml_nosafe(path: str) -> dict:
    with open(path, "r") as file:
        data = yaml.load(file)
    return data


def combine_dict(dict_list: List[dict]) -> dict:
    result = {}
    for dict_single in dict_list:
        result.update(dict_single)
    return result


def rmse_loss(predictions, targets):
    mse = F.mse_loss(predictions, targets)
    rmse = torch.sqrt(mse)
    return rmse


def convert_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def convert_numpy_to_tensor(arr: np.ndarray, device=None) -> torch.Tensor:
    return torch.from_numpy(arr).to(device) if device else torch.from_numpy(arr)


def split_dataset(dataset: Dataset, split_ratio: float) -> Tuple[Dataset]:
    r"""Splits the dataset with the given ```split ratio```

    Args:
        dataset (Dataset): Dataset to split
        split_ratio (float): split ratio train and valid

    Returns:
        Tuple[Dataset]: Tuple of train and valid set
    """

    val_len = int(split_ratio * len(dataset))
    train_len = len(dataset) - val_len

    train_set, val_set = random_split(dataset, [train_len, val_len])
    return train_set, val_set


def save_config(path: str, data: dict) -> None:
    with open(path, "w") as file:
        yaml.dump(data, file)


def mean_std(loader):
    images, _, _, _ = next(iter(loader))
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    return mean, std

def mean_std_speed(loader):
    _, speed, _, _ = next(iter(loader))
    # shape of images = [b,c,w,h]
    mean, std = speed.mean([0]), speed.std([0])
    return mean, std

