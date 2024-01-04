import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from autobrains.data_loader.data_classes import FrameData, VideoRecord
from autobrains.utils.utils import make_video_record_list


class VideoFrameDataset(torch.utils.data.Dataset):
    r"""
    A highly efficient and adaptable dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors where FRAMES=x if the ``ImglistToTensor()``
    transform is used.

    Args:
        root_path: The root path in which video folders lie.
                   this is ROOT_DATA from the description above.

    """

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
        """Single frame dataloader

        Args:
            record (FrameData): _description_

        Returns:
            _type_: _description_
        """

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


# class ImglistToTensor(torch.nn.Module):
#     """
#     Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
#     of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
#     Can be used as first transform for ``VideoFrameDataset``.
#     """

#     @staticmethod
#     def forward(
#         img_list: List[Image.Image],
#     ) -> "torch.Tensor[NUM_IMAGES, CHANNELS, HEIGHT, WIDTH]":
#         """
#         Converts each PIL image in a list to
#         a torch Tensor and stacks them into
#         a single tensor.

#         Args:
#             img_list: list of PIL images.
#         Returns:
#             tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
#         """
#         return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
