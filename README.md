# autobrains


```python
-> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple[
            "torch.Tensor[num_frames, channels, height, width]", Union[int, List[int]]
        ],
        Tuple[Any, Union[int, List[int]]],
    ]


-> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple[
            "torch.Tensor[num_frames, channels, height, width]", Union[int, List[int]]
        ],
        Tuple[Any, Union[int, List[int]]],
    ]
```

TODOs:
[] logger
    [x] save checkpoint
    [x] save config
    [] plot: [x] loss & [x] trajectory (full & segments)
[] tensorboard integration
[] setup.py
    [] requirements
[] Readme.md


https://stackoverflow.com/questions/582336/how-do-i-profile-a-python-script


# References
https://github.com/HumanCompatibleAI/imitation?tab=readme-ov-file

# Mean & Std

```bash
Image:
mean tensor([0.4452, 0.4627, 0.4797]) 
std:  tensor([0.1956, 0.2045, 0.2308])

Speed:
mean: tensor([6.4771]) 
std:  tensor([3.7553])
```