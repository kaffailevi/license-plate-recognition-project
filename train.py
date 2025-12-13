import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import CarPlateDataset, collate_fn
from model_utils import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a license plate detector with PyTorch."
    )
    parser.add_argument("--data-dir", required=True, help="Path to extracted dataset.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--output",
        default="checkpoints/model.pt",
        help="Where to store the trained model weights.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = CarPlateDataset(
        args.data_dir, train=True, split_ratio=0.8, seed=args.seed
    )
    val_dataset = CarPlateDataset(
        args.data_dir, train=False, split_ratio=0.8, seed=args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = create_model(num_classes=2, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        avg_loss = running_loss / len(train_loader)
        lr_scheduler.step()

        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            for images, _ in val_loader:
                images = [img.to(device) for img in images]
                _ = model(images)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved weights to {args.output}")


if __name__ == "__main__":
    main()
