from torchvision import transforms

transform_resnet = transforms.Compose(
    [
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4452, 0.4627, 0.4797], std=[0.1956, 0.2045, 0.2308]
        ),
    ]
)

transform_speed = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[6.4771], std=[3.7553]),
    ]
)
