import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from autobrains.data_loader.data_classes import FrameData, VideoRecord
from autobrains.utils.utils import make_video_record_list


class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_datapath: str,
        frames_per_segment: int = 1,
        imagefile_template: str = "img_{:05d}.jpg",
        transform: List = None,
        test_mode: bool = False,
    ):
        super(VideoFrameDataset, self).__init__()

        self._root_datapath = root_datapath

        self._get_video_instances()
        self._load_metadata()

        # load data
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.image_transform = transform[0]

        self.speed_transform = None
        if len(transform) > 1:
            self.speed_transform = transform[1]

        self.test_mode = test_mode

    @property
    def video_instances(self):
        return self._video_instances

    @property
    def video_records(self):
        return self._video_records

    def _get_video_instances(self):
        waypoints_path = os.path.join(self._root_datapath, "waypoints")
        self._video_instances = os.listdir(waypoints_path)

    def _load_image(self, image_filename: str) -> Image.Image:
        return Image.open(image_filename)

    def _load_metadata(self):
        self._video_records: List[FrameData] = []

        for video_instance in self._video_instances:
            video_record = VideoRecord(
                root_datapath=self._root_datapath, video_instance=video_instance
            )

            self._video_records += make_video_record_list(video_record)

    def __getitem__(self, idx: int):
        record: FrameData = self._video_records[idx]
        return self._get(record)

    def _get(self, record: FrameData) -> FrameData:
        
        # load self.frames_per_segment = 1 consecutive frames
        for _ in range(self.frames_per_segment):
            record.rgb = self._load_image(record.rgb_path)

        if self.image_transform is not None:
            record.rgb = self.image_transform(record.rgb)

        if self.speed_transform is not None:
            speed = torch.tensor(record.speed, dtype=torch.float32)
            mean = 6.4771
            std = 3.7553
            speed = (speed - mean) / std
        else:
            speed = torch.tensor(record.speed, dtype=torch.float32)

        waypoints = torch.tensor(record.waypoints, dtype=torch.float32)

        return record.rgb, speed, waypoints, record.instance

    def __len__(self):
        return len(self.video_records)
