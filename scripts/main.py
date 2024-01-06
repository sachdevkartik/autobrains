import argparse
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from autobrains.data_loader.augmentations import transform_resnet
from autobrains.data_loader.video_loader import VideoFrameDataset
from autobrains.models.cnnlstm import CNNLSTMBaseline2
from autobrains.models.vit_rnn import VitRNNBaseline
from autobrains.trainer.behaviour_cloner import BehaviourCloner
from autobrains.utils.utils import *

FILE_PATH = os.path.abspath(__file__)


def get_model(config: dict, model_name: str) -> torch.nn.Module:
    """Loads models depending as per the name and config

    Args:
        config (dict): model config
        model_name (str): short-hand name of the model. Choices : {"cnn", "cvt" or "levit"}

    Returns:
        torch.nn.Module: CNN or ViT Baseline model
    """
    if model_name == "cnn":
        return CNNLSTMBaseline2(config)
    if model_name == "cvt" or "levit":
        return VitRNNBaseline(config)


def init_logger_dir(logger_path: str, checkpoint: bool = False) -> None:
    """Initialize logging directory

    Args:
        logger_path (str): path of logging dir
        checkpoint (bool, optional): To save checkpoint or not. ```Default = True```
    """

    # make root logger
    os.makedirs(logger_path, exist_ok=True)

    # make plot dir
    directory = os.path.join(logger_path, "plots")
    os.makedirs(directory, exist_ok=True)

    # make checkpoint dir
    if checkpoint:
        directory = os.path.join(logger_path, "checkpoint")
        os.makedirs(directory, exist_ok=True)


def load_baseline_config(model_config: str, common_config: str) -> dict:
    """Loads config and returns config dict

    Args:
        model_config (str): Model e.g. CNN, CvT, LeViT details are there
        common_config (str): Config common to all the models

    Returns:
        dict: Dict of Config
    """

    # load configs
    baseline_config_path = make_absolute_path(os.path.abspath(__file__), model_config)
    common_config_path = make_absolute_path(os.path.abspath(__file__), common_config)

    config_model = load_yaml(baseline_config_path)
    config_common = load_yaml(common_config_path)

    # temp solution
    config_model["model"]["num_waypoints"] = config_common["num_waypoints"]
    config_model["model"]["dim_waypoints"] = config_common["dim_waypoints"]

    config = combine_dict([config_common, config_model])
    return config


def main(args):
    # config boiler plate
    config = load_baseline_config(
        common_config=args.common_config, model_config=args.model_config
    )
    root_datapath = config["root_datapath"]
    logger_path = config["logger_path"]
    device = config["device"]
    split_ratio = config["split_ratio"]

    batch_size = config["train"]["batch_size"]
    learning_rate = config["train"]["learning_rate"]
    config["train"]["device"] = config["device"]

    # logging boiler plate
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    root_datapath = os.path.join(os.path.dirname(FILE_PATH), root_datapath)
    logger_path = os.path.join(os.path.dirname(FILE_PATH), logger_path, current_time)
    checkpoint_path = os.path.join(logger_path, "checkpoint", "weights.pth")
    plots_path = os.path.join(logger_path, "plots")
    init_logger_dir(logger_path, checkpoint=True)

    # config boiler plate
    config["train"]["checkpoint_path"] = checkpoint_path
    config["train"]["plots_path"] = plots_path
    config["logger_path"] = logger_path
    train_config = config["train"]

    # load model
    model = get_model(config["model"], model_name=args.model)
    model.to(device)

    # load optimizers, criterion
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # load dataset
    video_frame_dataset = VideoFrameDataset(
        root_datapath=root_datapath, transform=[transform_resnet]
    )

    # split dataset
    train_set, val_set = split_dataset(video_frame_dataset, split_ratio=split_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # init trainer
    behaviour_cloner = BehaviourCloner(
        config=train_config,
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        use_scheduler=train_config["use_scheduler"],
    )

    # save config
    config_file = os.path.join(logger_path, "config.yaml")
    save_config(config_file, config)

    # train & plot loss
    behaviour_cloner.train()
    behaviour_cloner.plot_loss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--common_config",
        default="../config/common.yaml",
        help="Path to the common config file",
    )
    parser.add_argument("--model_config", help="Path to the common config file")
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "cvt", "levit"],
        default="cnn",
        help="Model Name",
    )

    args = parser.parse_args()

    main(args)
