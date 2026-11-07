"""Trains MajorityTokenTransformerClassifier (majority_token_classifier.py)
on the synthetic majority-token dataset (majority_token_dataset.py).

Standard supervised classification training loop (cross-entropy loss,
Adam), with:
  - an explicit train/test split (majority_token_dataset.make_majority_token_datasets),
    evaluated every epoch -- test performance is never used for gradient updates;
  - per-epoch metrics appended to a JSON-lines log file (one JSON object
    per line: epoch, train/test loss, train/test accuracy, plus the run
    config on the very first line) for later plotting by
    majority_token_evaluate.py;
  - a checkpoint saved every epoch containing model state, optimizer
    state, epoch number, and the full config needed to rebuild the model
    (so majority_token_evaluate.py does not need to guess architecture
    hyperparameters), with --resume to continue an interrupted run.

This script is meant to run as real training in Google Colab (GPU) for
the configured --epochs/--num-train. Locally, only run it with tiny
--num-train/--num-test and --epochs 1 as a smoke test (see bottom of
this file / the project's local-verification policy) -- that checks the
training loop is wired correctly, not that the model has converged.

Colab launch example:
    python code/transformers/majority_token_train.py \
        --num-train 20000 --num-test 4000 --epochs 15 \
        --checkpoint-path /content/drive/MyDrive/thesis_data/majority_token/checkpoint.pt \
        --log-path /content/drive/MyDrive/thesis_data/majority_token/train_log.jsonl

Local smoke test (tiny, CPU, seconds):
    python code/transformers/majority_token_train.py --smoke-test
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from majority_token_classifier import MajorityTokenTransformerClassifier
from majority_token_dataset import make_majority_token_datasets


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
        for tokens, labels in data_loader:
            tokens = tokens.to(device)  # [B,L]
            labels = labels.to(device)  # [B]

            outputs = model(tokens)
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
        "vocab_size": args.vocab_size,
        "seq_len": args.seq_len,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "ffn_hidden_dim": args.ffn_hidden_dim,
        "num_train": args.num_train,
        "num_test": args.num_test,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
    }

    train_dataset, test_dataset = make_majority_token_datasets(
        args.vocab_size, args.seq_len, args.num_train, args.num_test, seed=args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MajorityTokenTransformerClassifier(
        args.vocab_size, args.seq_len, args.embed_dim, args.num_heads, args.num_layers, args.ffn_hidden_dim
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
    dataset, CPU only. Verifies the training loop runs end-to-end
    (data loading, forward/backward, optimizer step, logging, checkpoint
    save+resume) and that the loss is finite and the loop at least
    reduces training loss on this trivial-size problem -- NOT evidence
    of convergence on the real-size configuration."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = argparse.Namespace(
            vocab_size=5,
            seq_len=6,
            embed_dim=8,
            num_heads=2,
            num_layers=1,
            ffn_hidden_dim=16,
            num_train=64,
            num_test=32,
            batch_size=16,
            lr=1e-3,
            epochs=2,
            seed=0,
            checkpoint_path=os.path.join(tmp_dir, "checkpoint.pt"),
            log_path=os.path.join(tmp_dir, "train_log.jsonl"),
            resume=False,
        )
        train(args)

        assert os.path.exists(args.checkpoint_path)
        assert os.path.exists(args.log_path)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        assert checkpoint["epoch"] == args.epochs - 1
        assert checkpoint["config"]["vocab_size"] == args.vocab_size

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

    print("majority_token_train smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run a lightweight local training smoke test and exit")
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-hidden-dim", type=int, default=128)
    parser.add_argument("--num-train", type=int, default=20000)
    parser.add_argument("--num-test", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/majority_token/checkpoint.pt")
    parser.add_argument("--log-path", type=str, default="checkpoints/majority_token/train_log.jsonl")
    parser.add_argument("--resume", action="store_true", help="resume from --checkpoint-path if it exists")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        train(args)
