"""Train the 'sublingual' head (3 classes for sublingual veins)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

from ai.training.dataset import TongueDataset
from ai.training.evaluate import evaluate


SUBLINGUAL_LABELS = ["怒張", "曲張", "囊柱囊泡"]


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model B (sublingual head).")
    parser.add_argument("--labels-json", required=True, type=Path)
    parser.add_argument("--img-dir", required=True, type=Path)
    parser.add_argument("--weights-out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    full_data = json.loads(args.labels_json.read_text(encoding="utf-8"))
    labels_map = {label: i for i, label in enumerate(SUBLINGUAL_LABELS)}

    valid_data = []
    for item in full_data:
        anns = item.get("annotations") or []
        if not anns:
            continue
        results = anns[0].get("result") or []
        if not results:
            continue
        choices = results[0].get("value", {}).get("choices") or []
        if not choices:
            continue
        if choices[0] in SUBLINGUAL_LABELS:
            valid_data.append(item)

    print(f"Filtered {len(valid_data)} sublingual records.")
    train_data, val_data = train_test_split(valid_data, test_size=0.2, random_state=42)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = TongueDataset(train_data, str(args.img_dir), labels_map, train_transform)
    val_ds = TongueDataset(val_data, str(args.img_dir), labels_map, val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = _autodetect_device()
    print(f"Device: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(SUBLINGUAL_LABELS))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_ds)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_acc = 100.0 * correct / total if total else 0.0
        print(f"Epoch {epoch+1:02d}/{args.epochs} - Loss {epoch_loss:.4f} | ValAcc {epoch_acc:.2f}%")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), args.weights_out)
            print(f"  saved best to {args.weights_out}")

    model.load_state_dict(torch.load(args.weights_out, map_location=device))
    report, kappa = evaluate(model, val_loader, SUBLINGUAL_LABELS, device)
    print("\n=== Classification Report ===")
    print(report)
    print(f"Cohen's Kappa: {kappa:.4f}")


if __name__ == "__main__":
    main()
