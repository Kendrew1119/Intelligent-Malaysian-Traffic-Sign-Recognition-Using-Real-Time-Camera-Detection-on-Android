"""Train a small seven-value classifier for cropped speed-limit signs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "speed_limit_reader" / "crops"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "speed_limit_reader.pt"
VALUES = ["5", "15", "30", "40", "50", "60", "80"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SpeedCropDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for class_index, value in enumerate(VALUES):
            class_dir = root / value
            if not class_dir.is_dir():
                continue
            self.samples.extend(
                (path, class_index)
                for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        if not self.samples:
            raise ValueError(f"No crop images found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


class SpeedLimitCNN(nn.Module):
    def __init__(self, classes: int = len(VALUES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            nn.MaxPool2d(2),
            self._block(32, 64),
            nn.MaxPool2d(2),
            self._block(64, 112),
            nn.MaxPool2d(2),
            self._block(112, 160),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(160, classes),
        )

    @staticmethod
    def _block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalise = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    training = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.28, contrast=0.28, saturation=0.18)],
                p=0.65,
            ),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.07, 0.07),
                scale=(0.82, 1.15),
                shear=4,
                fill=238,
            ),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.18, fill=238),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.15, 1.4))], p=0.28),
            transforms.ToTensor(),
            normalise,
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalise,
        ]
    )
    return training, evaluation


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(labels.numpy())
    probs = np.concatenate(probabilities)
    truth = np.concatenate(targets)
    accuracy = float((probs.argmax(axis=1) == truth).mean())
    return accuracy, probs, truth


def confusion_matrix(probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(VALUES), len(VALUES)), dtype=np.int64)
    for truth, prediction in zip(targets, probabilities.argmax(axis=1)):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def choose_threshold(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    candidates: list[dict[str, float]] = []
    for threshold in np.arange(0.50, 0.951, 0.01):
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        accuracy = float(correct[accepted].mean()) if accepted.any() else 0.0
        candidates.append(
            {"threshold": round(float(threshold), 2), "coverage": coverage, "accuracy": accuracy}
        )
    safe = [item for item in candidates if item["accuracy"] >= 0.95 and item["coverage"] >= 0.35]
    if safe:
        selected = max(safe, key=lambda item: (item["coverage"], item["accuracy"]))
    else:
        eligible = [item for item in candidates if item["coverage"] >= 0.35]
        selected = max(eligible, key=lambda item: (item["accuracy"], item["coverage"]))
    selected = dict(selected)
    selected["threshold"] = max(0.70, selected["threshold"])
    return selected


def write_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *VALUES])
        for value, row in zip(VALUES, matrix.tolist()):
            writer.writerow([value, *row])


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 48:
        raise ValueError("Use positive epochs/batch size and image size of at least 48.")
    set_seed(args.seed)
    device = choose_device(args.device)
    train_transform, evaluation_transform = make_transforms(args.image_size)
    dataset_root = args.dataset.resolve()
    train_set = SpeedCropDataset(dataset_root / "train", train_transform)
    valid_set = SpeedCropDataset(dataset_root / "valid", evaluation_transform)
    test_set = SpeedCropDataset(dataset_root / "test", evaluation_transform)

    training_counts = Counter(label for _, label in train_set.samples)
    sample_weights = [1.0 / training_counts[label] for _, label in train_set.samples]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SpeedLimitCNN().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.04)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    best_loss = math.inf
    patience_left = args.patience
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    print(
        f"Training {len(train_set)} crops; validating {len(valid_set)}; "
        f"testing {len(test_set)} on {device}."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimiser.step()
            running_loss += float(loss.item()) * labels.shape[0]
            seen += labels.shape[0]
            correct += int((logits.argmax(dim=1) == labels).sum().item())
        scheduler.step()
        train_loss = running_loss / seen
        train_accuracy = correct / seen
        valid_accuracy, _, _ = evaluate(model, valid_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "valid_accuracy": valid_accuracy,
                "learning_rate": optimiser.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch {epoch:02d}: loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.3f} valid_acc={valid_accuracy:.3f}"
        )
        improved = valid_accuracy > best_accuracy or (
            math.isclose(valid_accuracy, best_accuracy) and train_loss < best_loss
        )
        if improved:
            best_accuracy = valid_accuracy
            best_loss = train_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    valid_accuracy, valid_probabilities, valid_targets = evaluate(model, valid_loader, device)
    threshold_result = choose_threshold(valid_probabilities, valid_targets)
    threshold = threshold_result["threshold"]
    test_accuracy, test_probabilities, test_targets = evaluate(model, test_loader, device)
    test_confidence = test_probabilities.max(axis=1)
    test_correct = test_probabilities.argmax(axis=1) == test_targets
    accepted = test_confidence >= threshold
    accepted_accuracy = float(test_correct[accepted].mean()) if accepted.any() else 0.0
    coverage = float(accepted.mean())

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.zeros(1, 3, args.image_size, args.image_size)
    traced = torch.jit.trace(model, example)
    torch.jit.save(traced, str(output_path))

    report_dir = output_path.parent / "speed_limit_reader_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_confusion_csv(
        report_dir / "test_confusion_matrix.csv",
        confusion_matrix(test_probabilities, test_targets),
    )

    manifest = {
        "model": "SpeedLimitCNN",
        "purpose": "Read the numeric value after YOLO detects any speed-limit class.",
        "values": VALUES,
        "image_size": args.image_size,
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "minimum_crop_side": 40,
        "confidence_threshold": threshold,
        "training_crops": len(train_set),
        "validation_crops": len(valid_set),
        "test_crops": len(test_set),
        "best_validation_accuracy": valid_accuracy,
        "test_accuracy": test_accuracy,
        "test_accepted_accuracy": accepted_accuracy,
        "test_coverage": coverage,
        "training_seconds": time.perf_counter() - started,
        "seed": args.seed,
        "caution": "The exported test split contains few examples for several values; camera validation is still required.",
    }
    manifest_path = output_path.with_name("speed_limit_reader_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
