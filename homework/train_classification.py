"""
train_classification.py
────────────────────────
Part 1 training script for the SuperTuxKart image classifier.

Run from the homework root directory:
    python3 train_classification.py

Design decisions explained inline.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# adjust import path if running from project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from homework.models import Classifier, save_model, calculate_model_size_mb
from homework.metrics import AccuracyMetric
from homework.datasets.classification_dataset import load_data


def train(
    # ── data ──────────────────────────────────
    train_path: str = "classification_data/train",
    val_path:   str = "classification_data/val",
    # ── training hyper-params ─────────────────
    epochs:     int   = 30,
    lr:         float = 1e-3,
    batch_size: int   = 128,
    weight_decay: float = 1e-4,
    # ── misc ──────────────────────────────────
    log_dir:    str = "logs/classification",
    num_workers: int = 2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    # "aug" pipeline applies RandomHorizontalFlip for training.
    # Validation uses "default" (no augmentation) to get a clean accuracy signal.
    train_loader = load_data(
        train_path,
        transform_pipeline="aug",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = load_data(
        val_path,
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = Classifier(in_channels=3, num_classes=6).to(device)
    print(f"Model size: {calculate_model_size_mb(model):.2f} MB")

    # ── Optimizer + Scheduler ───────────────────────────────────────────
    # AdamW: Adam with decoupled weight decay (better regularisation than L2 in Adam).
    # OneCycleLR: warms up then decays LR following a cosine schedule.
    #   - peak_lr reached at ~30% of training, then anneals back.
    #   - Typically converges faster and to a better minimum than constant LR.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    # CrossEntropyLoss = LogSoftmax + NLLLoss; numerically stable.
    # label_smoothing=0.1 prevents overconfident predictions → better generalisation.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Logging ──────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir)
    metric = AccuracyMetric()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Training phase ────────────────────────────────────────────
        model.train()  # activates BatchNorm training stats + Dropout
        metric.reset()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()

            # Gradient clipping: prevents exploding gradients; max norm = 1.0
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            metric.add(logits.argmax(dim=1), labels)
            total_loss += loss.item()

        train_metrics = metric.compute()
        avg_loss = total_loss / len(train_loader)

        # ── Validation phase ──────────────────────────────────────────
        model.eval()   # disables BatchNorm update + Dropout
        metric.reset()

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                preds = model.predict(images)
                metric.add(preds, labels)

        val_metrics = metric.compute()
        val_acc = val_metrics["accuracy"]

        # Logging
        writer.add_scalar("Loss/train",        avg_loss,                  epoch)
        writer.add_scalar("Accuracy/train",    train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val",      val_acc,                   epoch)
        writer.add_scalar("LR",                scheduler.get_last_lr()[0], epoch)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss {avg_loss:.4f} | "
            f"Train acc {train_metrics['accuracy']:.3f} | "
            f"Val acc {val_acc:.3f}"
        )

        # ── Save best checkpoint ──────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_model(model)
            print(f"  ✓ Saved best model → {save_path}  (val acc={val_acc:.3f})")

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",   default="classification_data/train")
    parser.add_argument("--val_path",     default="classification_data/val")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_dir",      default="logs/classification")
    parser.add_argument("--num_workers",  type=int,   default=2)
    args = parser.parse_args()

    train(**vars(args))
