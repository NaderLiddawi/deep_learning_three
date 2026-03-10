"""
Surgical revision of train_detection.py.
Main changes:
1. Adds lane-aware weighting to the depth loss, because grading cares about lane depth.
2. Keeps the same training API and overall structure.
3. Saves the best checkpoint using IoU first, then depth as a tiebreaker.
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from homework.models import Detector, save_model, calculate_model_size_mb
from homework.metrics import DetectionMetric
from homework.datasets.road_dataset import load_data


def train(
    train_path: str = "drive_data/train",
    val_path: str = "drive_data/val",
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    depth_loss_weight: float = 1.0,
    lane_depth_boost: float = 2.0,
    seg_class_weight: list | None = None,
    log_dir: str = "logs/detection",
    num_workers: int = 2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = load_data(
        train_path,
        transform_pipeline="default",
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

    model = Detector(in_channels=3, num_classes=3).to(device)
    print(f"Model size: {calculate_model_size_mb(model):.2f} MB")

    if seg_class_weight is None:
        seg_class_weight = [1.0, 4.0, 4.0]
    class_weights = torch.tensor(seg_class_weight, device=device)

    seg_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    depth_loss_fn = nn.SmoothL1Loss(beta=0.1, reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    writer = SummaryWriter(log_dir)
    metric = DetectionMetric(num_classes=3)

    best_iou = -1.0
    best_depth = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        metric.reset()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_depth_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            depth_gt = batch["depth"].to(device)
            seg_gt = batch["track"].to(device).long()

            optimizer.zero_grad()
            seg_logits, depth_pred = model(images)

            l_seg = seg_loss_fn(seg_logits, seg_gt)

            depth_per_pixel = depth_loss_fn(depth_pred, depth_gt)
            lane_mask = (seg_gt > 0).float()
            depth_weights = 1.0 + lane_depth_boost * lane_mask
            l_depth = (depth_per_pixel * depth_weights).sum() / depth_weights.sum().clamp_min(1.0)

            loss = l_seg + depth_loss_weight * l_depth
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_seg_loss += l_seg.item()
            total_depth_loss += l_depth.item()

            seg_preds = seg_logits.argmax(dim=1)
            metric.add(seg_preds, seg_gt, depth_pred, depth_gt)

        train_m = metric.compute()
        n = len(train_loader)

        model.eval()
        metric.reset()
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device)
                depth_gt = batch["depth"].to(device)
                seg_gt = batch["track"].to(device).long()

                seg_preds, depth_pred = model.predict(images)
                metric.add(seg_preds, seg_gt, depth_pred, depth_gt)

        val_m = metric.compute()

        writer.add_scalar("Loss/train", total_loss / n, epoch)
        writer.add_scalar("Loss/seg", total_seg_loss / n, epoch)
        writer.add_scalar("Loss/depth", total_depth_loss / n, epoch)
        writer.add_scalar("IOU/train", train_m["iou"], epoch)
        writer.add_scalar("IOU/val", val_m["iou"], epoch)
        writer.add_scalar("DepthMAE/val", val_m["abs_depth_error"], epoch)
        writer.add_scalar("DepthMAE_lanes/val", val_m["tp_depth_error"], epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss {total_loss / n:.4f} "
            f"(seg={total_seg_loss / n:.4f}, depth={total_depth_loss / n:.4f}) | "
            f"Train IOU {train_m['iou']:.3f} | "
            f"Val IOU {val_m['iou']:.3f} | "
            f"Depth MAE {val_m['abs_depth_error']:.4f} | "
            f"Lane MAE {val_m['tp_depth_error']:.4f}"
        )

        should_save = (
            (val_m["iou"] > best_iou) or
            (abs(val_m["iou"] - best_iou) < 1e-6 and val_m["abs_depth_error"] < best_depth)
        )
        if should_save:
            best_iou = val_m["iou"]
            best_depth = val_m["abs_depth_error"]
            save_path = save_model(model)
            print(
                f"  ✓ Saved best model → {save_path} "
                f"(val IOU={val_m['iou']:.3f}, depth MAE={val_m['abs_depth_error']:.4f})"
            )

    writer.close()
    print("\nTraining complete.")
    print(f"  Best val IOU      : {best_iou:.3f}  (target > 0.75)")
    print(f"  Best val depth MAE: {best_depth:.4f}  (target < 0.05)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="drive_data/train")
    parser.add_argument("--val_path", default="drive_data/val")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--depth_loss_weight", type=float, default=1.0)
    parser.add_argument("--lane_depth_boost", type=float, default=2.0)
    parser.add_argument("--log_dir", default="logs/detection")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train(**vars(args))
