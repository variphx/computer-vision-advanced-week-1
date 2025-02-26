from typing import Literal
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as nn_functional
from torch.utils.data import DataLoader

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from torchvision.models.alexnet import alexnet, AlexNet_Weights
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize
from torchvision.io.image import decode_image, ImageReadMode

from datasets import load_dataset, Dataset, ClassLabel

CLASS_NAMES = [
    "manhole_cover",
    "pencil_box",
    "pillow",
    "traffic_signs",
]


def load_imagefolder(
    data_dir: str | Path = Path("augmentated_data"),
    mode: str | ImageReadMode = ImageReadMode.RGB,
    class_names: list[str] | None = CLASS_NAMES,
):
    if not isinstance(data_dir, Path):
        data_dir: Path = Path(data_dir)

    if class_names:
        for class_name in class_names:
            class_dir = data_dir / class_name
            for image_path in class_dir.iterdir():
                image_tensor = decode_image(str(image_path), mode=mode)
                yield {
                    "image": image_tensor,
                    "label": class_name,
                }

        return

    for class_dir in data_dir.iterdir():
        class_name = class_dir.name
        for image_path in class_dir.iterdir():
            image_tensor = decode_image(str(image_path), mode=mode)
            yield {
                "image": image_tensor,
                "label": class_name,
            }


# full_dataset_dict = (
#     Dataset.from_generator(load_imagefolder)
#     .cast_column("label", ClassLabel(names=CLASS_NAMES))
#     .train_test_split(test_size=0.4, stratify_by_column="label")
# )
full_dataset_dict = load_dataset(
    "imagefolder",
    data_dir="augmented_data",
    split="train",
).train_test_split(test_size=0.4, stratify_by_column="label")
test_dataset_dict = full_dataset_dict["test"].train_test_split(
    test_size=0.5, stratify_by_column="label"
)

train_dataset, val_dataset, test_dataset = (
    full_dataset_dict["train"].with_format("torch"),
    test_dataset_dict["train"].with_format("torch"),
    test_dataset_dict["test"].with_format("torch"),
)


def collate_fn(batch: list[dict[Literal["image", "label"], torch.Tensor]]):
    transforms = Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = torch.stack([transforms(x["image"]) for x in batch])
    label = torch.tensor([x["label"] for x in batch])

    return {"image": image, "label": label}


train_dataloader, val_dataloader, test_dataloader = (
    DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        num_workers=11,
        shuffle=True,
    ),
    DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        num_workers=11,
        shuffle=False,
    ),
    DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        num_workers=11,
        shuffle=False,
    ),
)


class AlexNetCustomized(LightningModule):
    def __init__(self, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        backbone = alexnet(weights=AlexNet_Weights.DEFAULT)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier

        self.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch: dict[Literal["image", "label"], torch.Tensor]):
        x, target = batch["image"], batch["label"]

        logits = self(x)
        loss = nn_functional.cross_entropy(logits, target)

        self.log("train_loss", loss.item())

        return loss

    @torch.no_grad()
    def validation_step(self, batch: dict[Literal["image", "label"], torch.Tensor]):
        x, target = batch["image"], batch["label"]

        logits = self(x)
        loss = nn_functional.cross_entropy(logits, target)

        self.log("val_loss", loss.item())

        return loss

    @torch.no_grad()
    def test_step(self, batch: dict[Literal["image", "label"], torch.Tensor]):
        x, target = batch["image"], batch["label"]

        logits = self(x)
        loss = nn_functional.cross_entropy(logits, target)

        self.log("test_loss", loss.item())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters())

        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=1,
                T_mult=2,
            ),
        }


model = AlexNetCustomized(num_classes=4)
model.requires_grad_(False)
model.classifier.requires_grad_(True)

callbacks = [EarlyStopping(monitor="val_loss")]
logger = CSVLogger(save_dir="logs", name="alexnet_customized")
trainer = Trainer(callbacks=callbacks, logger=logger)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

trainer.test(model=model, dataloaders=test_dataloader)
