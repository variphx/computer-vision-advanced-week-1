import csv
import shutil
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

with open(DATA_DIR / "trainLabels.csv", "r") as f:
    csv_read = csv.reader(f)
    next(csv_read)

    class_names = set()

    for _, class_name in csv_read:
        class_names.add(class_name)

random_class_indices = np.random.randint(low=0, high=len(class_names), size=(4,))
class_names = sorted(class_names)
class_names = set(class_names[index] for index in random_class_indices)

TRAIN_IMAGES_DIR = DATA_DIR / "train"
SELECTED_IMAGES_DIR = DATA_DIR / "selected"

if SELECTED_IMAGES_DIR.exists():
    SELECTED_IMAGES_DIR.rmdir()
    
SELECTED_IMAGES_DIR.mkdir()

with open(DATA_DIR / "trainLabels.csv", "r") as f:
    csv_read = csv.reader(f)
    next(csv_read)

    for image_id, class_name in csv_read:
        if class_name not in class_names:
            continue

        CLASS_DIR = SELECTED_IMAGES_DIR / class_name
        CLASS_DIR.mkdir(exist_ok=True)

        shutil.copy(TRAIN_IMAGES_DIR / f"{image_id}.png", CLASS_DIR / f"{image_id}.png")
