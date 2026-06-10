import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from model import StoneCNN
from synthetic import PATCH_SIZE, generate_batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_tensors(images, labels, device):
    data = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    target = torch.from_numpy(labels).long()
    return data.to(device), target.to(device)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            pred = logits.argmax(dim=1)
            correct += int((pred == batch_y).sum().item())
            total += batch_y.numel()
    return correct / max(total, 1)


def export_onnx(model, output_path, patch_size=PATCH_SIZE):
    model.eval()
    dummy = torch.randn(1, 3, patch_size, patch_size)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/models/stone_cls.onnx")
    parser.add_argument("--meta", default="/models/stone_cls.meta.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-class", type=int, default=3500)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cpu")

    images, labels = generate_batch(count_per_class=args.samples_per_class)
    split = int(len(images) * 0.9)
    train_x, train_y = prepare_tensors(images[:split], labels[:split], device)
    val_x, val_y = prepare_tensors(images[split:], labels[split:], device)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.batch_size)

    model = StoneCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        best_acc = max(best_acc, val_acc)
        print(f"epoch={epoch + 1} val_acc={val_acc:.4f}")

    export_onnx(model, args.output)
    meta = {
        "patchSize": PATCH_SIZE,
        "classes": ["empty", "black", "white"],
        "valAccuracy": round(best_acc, 4),
        "epochs": args.epochs,
    }
    with open(args.meta, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(f"saved {args.output} val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
