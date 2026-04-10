"""
Training script for the face embedding network.

Uses Triplet Margin Loss to learn an embedding space where faces of the
same person are pulled together and faces of different people are pushed
apart. The margin controls the minimum gap enforced between positive and
negative pairs.

Usage
-----
    python -m src.train --data_dir data/train --epochs 30 --batch_size 32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TripletFaceDataset, get_train_transforms
from src.model import FaceEmbeddingNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train face embedding network with triplet loss"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to training data (one sub-folder per identity)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument(
        "--save_path", type=str, default="models/face_embedding.pth",
        help="Where to save the best model weights",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device (cpu / cuda / mps). Auto-detected if omitted.",
    )
    return parser.parse_args()


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.TripletMarginLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for anchor, positive, negative in tqdm(loader, desc="  Training", leave=False):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * anchor.size(0)

    return running_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    dataset = TripletFaceDataset(
        root_dir=args.data_dir,
        transform=get_train_transforms(),
    )
    print(f"Loaded {len(dataset)} samples across {len(dataset.labels)} identities")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = FaceEmbeddingNet(embedding_dim=args.embedding_dim).to(device)
    criterion = nn.TripletMarginLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={epoch_loss:.4f}  "
              f"lr={lr:.2e}  time={elapsed:.1f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (loss={best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
