from tqdm.auto import tqdm
import json
from pathlib import Path
import torch
from torch import nn
from torchvision.models.alexnet import alexnet, AlexNet_Weights
from torchvision.io.image import decode_image, ImageReadMode
from torchvision.transforms.v2.functional import (
    to_dtype,
    resize_image,
    center_crop_image,
    normalize_image,
)

DATA_DIR = Path("data")
PARSED_DATA_DIR = Path("parsed_data")


class AlexNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        pretrained_alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.features = pretrained_alexnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        if x.dim() == 3:
            x = x.flatten(0)
        elif x.dim() == 4:
            x = x.flatten(1)
        return x


alexnet_extractor = AlexNetExtractor()

for class_dir in DATA_DIR.iterdir():
    class_name = class_dir.name

    parsed_class_dir = PARSED_DATA_DIR / class_name
    parsed_class_dir.mkdir()

    for image_path in tqdm(class_dir.iterdir(), desc=class_name):
        image_tensor = decode_image(image_path, mode=ImageReadMode.RGB)
        image_tensor = resize_image(image_tensor, [256])
        image_tensor = center_crop_image(image_tensor, [224])
        image_tensor = to_dtype(image_tensor, scale=True)
        image_tensor = normalize_image(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        with torch.no_grad():
            features_vector: torch.Tensor = alexnet_extractor(image_tensor)

        with open(parsed_class_dir / f"{image_path.name}.json", "w") as f:
            json.dump(features_vector.tolist(), f)
