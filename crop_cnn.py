from pathlib import Path

import numpy as np
import torch
from torch import nn


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 128
MODEL_PATH = Path(__file__).resolve().parent / "crop_cnn.pt"


class CropCNN(nn.Module):
    def __init__(self, crop_count, stage_count, health_count):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.crop_head = nn.Linear(128, crop_count)
        self.stage_head = nn.Linear(128, stage_count)
        self.health_head = nn.Linear(128, health_count)

    def forward(self, x):
        x = self.shared(self.features(x))
        return self.crop_head(x), self.stage_head(x), self.health_head(x)


class CropGroupCNN(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, class_count),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def image_to_tensor(image, image_size=IMAGE_SIZE):
    image = image.convert("RGB").resize((image_size, image_size))
    data = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (data - mean) / std


def classify_folder(crop, folder_name):
    normalized = folder_name.lower()
    if crop == "cucumber":
        if "unhealthy" in normalized:
            return "Leaves", "unhealthy"
        if "healthy" in normalized:
            return "Leaves", "healthy"
        return folder_name, "healthy"

    if "wilted" in normalized:
        return "Bloom", "unhealthy"
    if "healthy" in normalized:
        return "Bloom", "healthy"
    return folder_name.replace("EarlyBloom", "Early Bloom").replace("MatureBud", "Mature Bud").replace("YoungBud", "Young Bud"), "healthy"


def list_image_samples(root):
    root = Path(root)
    datasets = [
        ("cucumber", root / "archive"),
        ("sunflower", root / "Sunflower Compressed"),
    ]
    samples = []
    for crop, folder in datasets:
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        for class_dir in sorted(folder.iterdir()):
            if not class_dir.is_dir():
                continue
            paths = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and not p.name.startswith("._")]
            if not paths:
                continue
            stage, health = classify_folder(crop, class_dir.name)
            for path in paths:
                samples.append((path, crop, stage, health))
    return samples
