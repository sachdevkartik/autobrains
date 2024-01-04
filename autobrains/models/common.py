import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101


def get_resnet(model_name, pretrained=False):
    if model_name == "resnet101":
        return resnet101(pretrained=pretrained)

    elif model_name == "resnet50":
        return resnet50(pretrained=pretrained)

    elif model_name == "resnet34":
        return resnet34(pretrained=pretrained)

    elif model_name == "resnet18":
        return resnet18(pretrained=pretrained)

    else:
        return resnet50(pretrained=False)


def get_activation(activation, **kwargs):
    if activation == "relu":
        return nn.ReLU(**kwargs)
    elif activation == "tanh":
        return nn.Tanh(**kwargs)
    elif activation == "sigmoid":
        return nn.Sigmoid(**kwargs)
    elif activation == "elu":
        return nn.ELU(**kwargs)
    elif activation == "selu":
        return nn.SELU(**kwargs)
    elif activation == "swish":
        return nn.SiLU(**kwargs)
    elif activation == "gelu":
        return nn.GELU(**kwargs)
    elif activation == "softplus":
        return nn.Softplus(**kwargs)
    else:
        return nn.Identity()


def get_sequential_mlp(
    input_size, units, activation, norm_func_name=None, need_norm=True
):
    in_size = input_size
    layers = []
    need_norm = need_norm
    for unit in units:
        layers.append(nn.Linear(in_size, unit))
        layers.append(get_activation(activation))
        in_size = unit
        if not need_norm:
            continue
        if norm_func_name is not None:
            need_norm = False
        if norm_func_name == "layer_norm":
            layers.append(torch.nn.LayerNorm(unit))
        elif norm_func_name == "batch_norm":
            layers.append(torch.nn.BatchNorm1d(unit))

    return nn.Sequential(*layers)
