import os
from typing import List, Union, Tuple, Any, Dict

from dataclasses import dataclass


@dataclass
class FrameData:
    rgb_path: str = None
    rgb = None
    speed: float = None
    waypoints: List[float] = None
    instance: str = None
    frame_idx: int = None


class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder of the videos.
    """

    def __init__(self, root_datapath: str, video_instance: str):
        self._root_datapath = root_datapath
        self._video_instance = video_instance
        self._rgb_base_path = os.path.join(self._root_datapath, "rgb", video_instance)
        self._waypoints_path = os.path.join(
            self._root_datapath, "waypoints", video_instance, "waypoints.npy"
        )
        self._speed_path = os.path.join(
            self._root_datapath, "speed", video_instance, "speed.npy"
        )

        self._generate_rgb_path_list()

    @property
    def video_instance(self) -> str:
        return self._video_instance

    @property
    def root_datapath(self) -> str:
        return self._root_datapath

    @property
    def rgb_base_path(self) -> str:
        return self._rgb_base_path

    @property
    def waypoints_path(self) -> str:
        return self._waypoints_path

    @property
    def speed_path(self) -> str:
        return self._speed_path

    @property
    def rgb_path_list(self) -> List[str]:
        return self._rgb_path_list

    def _generate_rgb_path_list(self):
        rgb_files = sorted(os.listdir(self._rgb_base_path))
        self._rgb_path_list = [
            os.path.join(self._rgb_base_path, rgb_file) for rgb_file in rgb_files
        ]

    def __str__(self):
        return f"\n speed_path: {self.speed_path} \n waypoints_path: {self.waypoints_path} \n rgb_path_list: {self.rgb_path_list}"
