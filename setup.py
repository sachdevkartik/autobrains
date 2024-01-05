from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="autobrains",
    version="0.1.0",
    license="MIT",
    description="A PyTorch-based library for Imitation Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kartik Sachdev",
    author_email="kartik.sachdev@rwth-aachen.de",
    url="git@github.com:sachdevkartik/autobrains.git",
    packages=find_packages(
        include=[
            "autobrains",
            "autobrains.data_loader",
            "autobrains.models",
            "autobrains.trainer",
            "autobrains.utils",
        ]
    ),
    keywords=[
        "Imitation Learning",
        "Deep Learning",
    ],
    install_requires=[
        "matplotlib>=3.2.2",
        "numpy",
        "pandas>=1.1.4",
        "seaborn>=0.11.0",
        "scikit-learn",
        "opencv-python>=4.1.1",
        "Pillow>=8.2.0",
        "PyYAML>=5.3.1",
        "requests>=2.25.1",
        "scipy>=1.4.1",
        "torch==2.0.1",
        "vit-pytorch==0.27.0",
        "torchvision>=0.7.0",
        "torchinfo",
        "tqdm>=4.41.0",
        "timm",
        "transformers>=4.18.0",
        "tensorboard>=2.4.1",
        "wandb",
        "ipython",
        "psutil",
        "thop",
        "albumentations>=1.0.3",
        "gdown",
        "split-folders",
        "ipywidgets",
        "einops",
        "protobuf==3.20.*",
        "lightly",
    ],
)
