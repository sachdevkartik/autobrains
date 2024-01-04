from typing import Any

from .levit import LeViT
from .cvt import CvT


def GetLeViT(num_classes: int, num_channels: int, img_size: int):
    """Wraps LeViT transformer architecture introduced in the paper: \n
    `LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference`

    Args:
        num_classes (int): # of classes for classification
        num_channels (int): # of channels of input image

    Returns:
        model (nn.Module): LeViT model

    https://openaccess.thecvf.com/content/ICCV2021/papers/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.pdf
    """

    model = LeViT(
        image_size=img_size,
        num_classes=num_classes,
        stages=3,  # number of stages
        dim=(64, 128, 128),  # dimensions at each stage
        depth=5,  # transformer of depth 4 at each stage
        heads=(2, 4, 5),  # heads at each stage
        mlp_mult=2,
        dropout=0.1,
        channels=num_channels,
    )
    return model


def VitModels(
    transformer_type: str,
    config: dict,
    last_layer_dim: int = 256,
    num_channels: int = 3,
    img_size: int = 224,
):
    """Get different transform architecture

    Args:
        transformer_type (str):
            name of the transformer ["CvT", "CCT", "TwinsSVT", "LeViT", "CaiT", "CrossViT", "PiT"]
        num_classes (int): # of classes for classification
        num_channels (int): # of channels of input image
        img_size (int): size of input image

    Returns:
        model (nn.Module): Required transformer architecture

    Example:
        >>>     TransformerModels(
        >>>     transformer_type="LeViT",
        >>>     num_channels=1,
        >>>     num_classes=3,
        >>>     img_size=224,
        >>>     {   "stages": 3,
        >>>         "dim": (64, 128, 128),
        >>>         "depth": 5,
        >>>         "heads": (2, 4, 5),
        >>>         "mlp_mult": 2,
        >>>         "dropout": 0.1})

    CvT: https://arxiv.org/pdf/2103.15808.pdf \n
    CCT: https://arxiv.org/pdf/2104.05704v4.pdf \n
    TwinsSVT: https://arxiv.org/pdf/2104.13840.pdf \n
    LeViT: https://openaccess.thecvf.com/content/ICCV2021/papers/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.pdf \n
    CaiT: https://arxiv.org/pdf/2103.17239.pdf \n
    CrossViT: https://arxiv.org/pdf/2103.14899.pdf \n
    PiT: https://arxiv.org/pdf/2103.16302.pdf \n
    Swin: https://arxiv.org/pdf/2103.14030.pdf \n
    T2TViT: https://arxiv.org/pdf/2101.11986v3.pdf \n
    """

    assert transformer_type in [
        "cvt",
        "CCT",
        "TwinsSVT",
        "levit",
        "CaiT",
        "CrossViT",
        "PiT",
        "Swin",
        "T2TViT",
        "CrossFormer",
    ]

    if transformer_type == "levit":
        model = LeViT(
            image_size=img_size,
            num_classes=last_layer_dim,
            channels=num_channels,
            stages=config["stages"],  # number of stages
            dim=tuple(config["dim"]),  # dimensions at each stage
            depth=config["depth"],  # transformer of depth 4 at each stage
            heads=tuple(config["heads"]),  # heads at each stage
            mlp_mult=config["mlp_mult"],
            dropout=config["dropout"],
        )

    elif transformer_type == "cvt":
        model = CvT(
            channels=num_channels,
            num_classes=last_layer_dim,
            s1_emb_dim=config["s1_emb_dim"],
            s1_emb_kernel=config["s1_emb_kernel"],
            s1_emb_stride=config["s1_emb_stride"],
            s1_proj_kernel=config["s1_proj_kernel"],
            s1_kv_proj_stride=config["s1_kv_proj_stride"],
            s1_heads=config["s1_heads"],
            s1_depth=config["s1_depth"],
            s1_mlp_mult=config["s1_mlp_mult"],
            s2_emb_dim=config["s2_emb_dim"],
            s2_emb_kernel=config["s2_emb_kernel"],
            s2_emb_stride=config["s2_emb_stride"],
            s2_proj_kernel=config["s2_proj_kernel"],
            s2_kv_proj_stride=config["s2_kv_proj_stride"],
            s2_heads=config["s2_heads"],
            s2_depth=config["s2_depth"],
            s2_mlp_mult=config["s2_mlp_mult"],
            s3_emb_dim=config["s3_emb_dim"],  
            s3_emb_kernel=config["s3_emb_kernel"],
            s3_emb_stride=config["s3_emb_stride"],
            s3_proj_kernel=config["s3_proj_kernel"],
            s3_kv_proj_stride=config["s3_kv_proj_stride"],
            s3_heads=config["s3_heads"],
            s3_depth=config["s3_depth"],
            s3_mlp_mult=config["s3_mlp_mult"],
            stages=config["stages"],
            mlp_last=config["mlp_last"],
            dropout=config["dropout"],
        )

    return model
