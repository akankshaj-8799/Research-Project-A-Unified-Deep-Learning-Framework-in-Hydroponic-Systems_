from pathlib import Path
import argparse
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from crop_cnn import CropGroupCNN, MODEL_PATH, image_to_tensor, list_image_samples


class CropImageDataset(Dataset):
    def __init__(self, samples, class_map, augment=False):
        self.samples = samples
        self.class_map = class_map
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, crop, stage, health = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tensor = image_to_tensor(image)
        label = self.class_map[(crop, stage, health)]
        return tensor, torch.tensor(label, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the crop image CNN.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def group_metrics(pred_indexes, true_indexes, class_labels):
    total = len(true_indexes)
    class_ok = crop_ok = stage_ok = health_ok = 0
    for pred_index, true_index in zip(pred_indexes, true_indexes):
        pred = class_labels[pred_index]
        true = class_labels[true_index]
        class_ok += pred == true
        crop_ok += pred[0] == true[0]
        stage_ok += pred[1] == true[1]
        health_ok += pred[2] == true[2]
    return class_ok / total, crop_ok / total, stage_ok / total, health_ok / total


def run_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    pred_indexes = []
    true_indexes = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred_indexes.extend(logits.detach().argmax(dim=1).cpu().tolist())
        true_indexes.extend(labels.detach().cpu().tolist())
    return total_loss / len(loader), pred_indexes, true_indexes


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    pred_indexes = []
    true_indexes = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        pred_indexes.extend(logits.argmax(dim=1).cpu().tolist())
        true_indexes.extend(labels.cpu().tolist())
    return total_loss / len(loader), pred_indexes, true_indexes


def main():
    args = parse_args()
    seed_everything()
    root = args.root.resolve()
    samples = list_image_samples(root)
    if not samples:
        raise RuntimeError(f"No training images found under {root}")

    class_labels = sorted({(crop, stage, health) for _, crop, stage, health in samples})
    class_map = {label: index for index, label in enumerate(class_labels)}
    stratify = [class_map[(crop, stage, health)] for _, crop, stage, health in samples]

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    train_dataset = CropImageDataset(train_samples, class_map, augment=True)
    val_dataset = CropImageDataset(val_samples, class_map)

    class_counts = {}
    for _, crop, stage, health in train_samples:
        class_counts[(crop, stage, health)] = class_counts.get((crop, stage, health), 0) + 1
    sample_weights = [1.0 / class_counts[(crop, stage, health)] for _, crop, stage, health in train_samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_samples), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = choose_device()
    model = CropGroupCNN(len(class_labels)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print("Classes:")
    for index, label in enumerate(class_labels):
        print(f"  {index}: {label}")

    best_score = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pred, train_true = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_pred, val_true = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        train_metrics = group_metrics(train_pred, train_true, class_labels)
        val_metrics = group_metrics(val_pred, val_true, class_labels)
        val_score = (val_metrics[1] + val_metrics[2] + val_metrics[3]) / 3
        if val_score > best_score:
            best_score = val_score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_class={val_metrics[0]:.3f} "
            f"val_crop={val_metrics[1]:.3f} "
            f"val_stage={val_metrics[2]:.3f} "
            f"val_health={val_metrics[3]:.3f}"
        )

    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_labels": class_labels,
            "crop_types": sorted({label[0] for label in class_labels}),
            "stages": sorted({label[1] for label in class_labels}),
            "healths": sorted({label[2] for label in class_labels}),
            "image_size": 128,
            "model_type": "group_cnn",
        },
        args.output,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
