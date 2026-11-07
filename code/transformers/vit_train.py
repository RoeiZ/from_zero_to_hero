"""Trains VisionTransformerClassifier (vit_patch_embedding.py) on CIFAR-10.

Standard supervised image-classification training loop (cross-entropy
loss, Adam), with:
  - an explicit train/test split -- CIFAR-10's own predefined train=True
    / train=False files (not a random re-split), evaluated every epoch,
    test performance never used for gradient updates;
  - per-epoch metrics appended to a JSON-lines log file (one JSON object
    per line: epoch, train/test loss, train/test accuracy, plus the run
    config on the very first line) for later plotting by vit_evaluate.py;
  - a checkpoint saved every epoch containing model state, optimizer
    state, epoch number, and the full config needed to rebuild the model
    (so vit_evaluate.py does not need to guess architecture
    hyperparameters), with --resume to continue an interrupted run.

Dataset caching (per CLAUDE.md): --data-root should point at a
persistent Google Drive path in Colab (e.g.
/content/drive/MyDrive/thesis_data/cifar10), not local ephemeral
/content storage, so CIFAR-10 is downloaded once and reused across
Colab sessions. torchvision.datasets.CIFAR10(download=True) already
skips re-downloading if the files exist at that root.

This script is meant to run as real training in Google Colab (GPU) for
the configured --epochs. Locally, only run --smoke-test, which trains
on a tiny synthetic CIFAR-10-shaped dataset ([B,3,32,32], 10 classes)
instead of downloading the real ~170MB dataset -- that checks the
training loop is wired correctly, not that the model has converged.

Colab launch example:
    python code/transformers/vit_train.py \
        --epochs 30 \
        --data-root /content/drive/MyDrive/thesis_data/cifar10 \
        --checkpoint-path /content/drive/MyDrive/thesis_data/vit_cifar10/checkpoint.pt \
        --log-path /content/drive/MyDrive/thesis_data/vit_cifar10/train_log.jsonl

Local smoke test (tiny synthetic data, CPU, seconds):
    python code/transformers/vit_train.py --smoke-test
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from vit_patch_embedding import VisionTransformerClassifier

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def make_cifar10_dataloaders(data_root: str, batch_size: int, num_workers: int = 2):
    """Downloads (if missing) and loads CIFAR-10's predefined train/test
    splits. See module docstring re: pointing data_root at persistent
    Drive storage in Colab.

    Returns:
        train_loader, test_loader: DataLoader, batches of (images [B,3,32,32], labels [B])
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def _make_synthetic_cifar10_like_dataloaders(num_train: int, num_test: int, batch_size: int, seed: int):
    """Tiny synthetic stand-in for CIFAR-10 (same tensor shapes, random
    values), so the training loop can be exercised locally without
    downloading the real dataset. NOT a dataset split in any meaningful
    sense -- exists only to test the loop wiring."""
    generator = torch.Generator().manual_seed(seed)
    train_images = torch.randn(num_train, 3, 32, 32, generator=generator)
    train_labels = torch.randint(0, 10, (num_train,), generator=generator)
    test_images = torch.randn(num_test, 3, 32, 32, generator=generator)
    test_labels = torch.randint(0, 10, (num_test,), generator=generator)

    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _run_one_epoch(model, data_loader, optimizer, device, train: bool):
    """Runs one full pass over data_loader. If train=True, updates model
    parameters; if train=False, runs in torch.no_grad() (evaluation only).

    Returns:
        mean_loss: float, cross-entropy averaged over all samples
        accuracy:  float, fraction of correctly predicted labels
    """
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in data_loader:
            images = images.to(device)  # [B,3,32,32]
            labels = labels.to(device)  # [B]

            outputs = model(images)
            logits = outputs["logits"]  # [B,num_classes]
            loss = nn.functional.cross_entropy(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += batch_size

    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return mean_loss, accuracy


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    config = {
        "dataset": "cifar10",
        "image_size": 32,
        "image_channels": 3,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "ffn_hidden_dim": args.ffn_hidden_dim,
        "num_classes": 10,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
    }

    if args.smoke_test:
        train_loader, test_loader = _make_synthetic_cifar10_like_dataloaders(
            args.num_train, args.num_test, args.batch_size, args.seed
        )
    else:
        train_loader, test_loader = make_cifar10_dataloaders(args.data_root, args.batch_size)

    model = VisionTransformerClassifier(
        config["image_size"], config["patch_size"], config["image_channels"], config["embed_dim"],
        config["num_heads"], config["num_layers"], config["ffn_hidden_dim"], config["num_classes"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {args.checkpoint_path} at epoch {start_epoch}")

    os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    log_mode = "a" if (args.resume and os.path.exists(args.log_path)) else "w"
    with open(args.log_path, log_mode) as log_file:
        if log_mode == "w":
            log_file.write(json.dumps({"config": config}) + "\n")

        for epoch in range(start_epoch, args.epochs):
            train_loss, train_acc = _run_one_epoch(model, train_loader, optimizer, device, train=True)
            test_loss, test_acc = _run_one_epoch(model, test_loader, optimizer, device, train=False)

            print(
                f"epoch {epoch:3d} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
                f"| test_loss {test_loss:.4f} test_acc {test_acc:.4f}"
            )

            log_file.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                    }
                )
                + "\n"
            )
            log_file.flush()

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": config,
                },
                args.checkpoint_path,
            )

    print(f"Checkpoint saved to {args.checkpoint_path}")
    print(f"Log saved to {args.log_path}")


def _smoke_test():
    """Lightweight local check: a couple of epochs on a tiny synthetic
    CIFAR-10-shaped dataset, CPU only, no download. Verifies the
    training loop runs end-to-end (data loading, forward/backward,
    optimizer step, logging, checkpoint save+resume) and that losses are
    finite -- NOT evidence of convergence on real CIFAR-10."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = argparse.Namespace(
            patch_size=4,
            embed_dim=8,
            num_heads=2,
            num_layers=1,
            ffn_hidden_dim=16,
            batch_size=16,
            lr=1e-3,
            epochs=2,
            seed=0,
            num_train=64,
            num_test=32,
            data_root=os.path.join(tmp_dir, "data"),
            checkpoint_path=os.path.join(tmp_dir, "checkpoint.pt"),
            log_path=os.path.join(tmp_dir, "train_log.jsonl"),
            resume=False,
            smoke_test=True,
        )
        train(args)

        assert os.path.exists(args.checkpoint_path)
        assert os.path.exists(args.log_path)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        assert checkpoint["epoch"] == args.epochs - 1
        assert checkpoint["config"]["patch_size"] == args.patch_size

        with open(args.log_path) as f:
            lines = [json.loads(line) for line in f]
        assert "config" in lines[0]
        assert len(lines) == args.epochs + 1
        assert all(torch.isfinite(torch.tensor(line["train_loss"])) for line in lines[1:])

        # Resume: train 1 more epoch from the saved checkpoint.
        args.epochs = 3
        args.resume = True
        train(args)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        assert checkpoint["epoch"] == 2

    print("vit_train smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run a lightweight local training smoke test (synthetic data) and exit")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ffn-hidden-dim", type=int, default=128)
    parser.add_argument("--num-train", type=int, default=64, help="only used by --smoke-test (synthetic data)")
    parser.add_argument("--num-test", type=int, default=32, help="only used by --smoke-test (synthetic data)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=str, default="data/cifar10")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/vit_cifar10/checkpoint.pt")
    parser.add_argument("--log-path", type=str, default="checkpoints/vit_cifar10/train_log.jsonl")
    parser.add_argument("--resume", action="store_true", help="resume from --checkpoint-path if it exists")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        train(args)
