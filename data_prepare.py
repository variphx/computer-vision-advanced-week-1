import shutil
from pathlib import Path
from tqdm.auto import tqdm

from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.v2 import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from torchvision.utils import save_image

transforms = Compose(
    [
        RandomResizedCrop((224, 224), antialias=True),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
    ]
)


def augmentate(image_path: str, multiply_coeff: int = 5):
    image = decode_image(image_path, mode=ImageReadMode.RGB)
    for _ in range(multiply_coeff):
        yield transforms(image)


def augmentate_data_dir(data_dir: str, augmentated_dir: str):
    data_dir: Path = Path(data_dir)
    augmentated_dir: Path = Path(augmentated_dir)

    if not augmentated_dir.exists():
        augmentated_dir.mkdir()

    for class_dir in data_dir.iterdir():
        class_name = class_dir.name

        augmentated_class_dir = augmentated_dir / class_name
        if augmentated_class_dir.exists():
            shutil.rmtree(augmentated_class_dir)
        augmentated_class_dir.mkdir()

        for image_path in tqdm(class_dir.iterdir(), desc=class_name):
            for i, augmented_image in enumerate(augmentate(str(image_path))):
                save_image(
                    augmented_image,
                    augmentated_class_dir
                    / f"{image_path.name[:image_path.name.find('.')]}_{i+1}.png",
                )


augmentate_data_dir("data", "augmented_data")
