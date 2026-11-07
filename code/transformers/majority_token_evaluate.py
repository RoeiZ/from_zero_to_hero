"""Loads a checkpoint saved by majority_token_train.py and produces the
qualitative + quantitative evaluation of MajorityTokenTransformerClassifier
on the held-out test split: training curves, overall/per-class test
accuracy, a confusion matrix, sample predictions, and -- the
visualization that is specific to a Transformer -- the last-layer
attention weights from the CLS token to every sequence position, which
should concentrate on the positions holding the true majority value.

No training happens here; this script only runs forward passes
(torch.no_grad()) against a checkpoint produced by majority_token_train.py.

Colab launch example (after majority_token_train.py has produced a
checkpoint + log on Drive):
    python code/transformers/majority_token_evaluate.py \
        --checkpoint-path /content/drive/MyDrive/thesis_data/majority_token/checkpoint.pt \
        --log-path /content/drive/MyDrive/thesis_data/majority_token/train_log.jsonl \
        --output-dir /content/drive/MyDrive/thesis_data/majority_token/figures
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from majority_token_classifier import MajorityTokenTransformerClassifier
from majority_token_dataset import make_majority_token_datasets


def load_model_and_config(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = MajorityTokenTransformerClassifier(
        config["vocab_size"],
        config["seq_len"],
        config["embed_dim"],
        config["num_heads"],
        config["num_layers"],
        config["ffn_hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


@torch.no_grad()
def compute_test_accuracy_and_confusion(model, test_dataset, device, batch_size=128):
    """
    Returns:
        accuracy:   float
        confusion:  [V,V] int64, confusion[true, pred] = count
    """
    num_classes = model.num_classes
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_correct = 0
    total_samples = 0
    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = model(tokens)["logits"]  # [B,V]
        preds = logits.argmax(dim=-1)  # [B]

        total_correct += (preds == labels).sum().item()
        total_samples += labels.shape[0]
        for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
            confusion[true_label, pred_label] += 1

    accuracy = total_correct / total_samples
    return accuracy, confusion


def plot_training_curves(log_path: str, save_path: str | None = None):
    """Reads the per-epoch JSONL log written by majority_token_train.py
    and plots train/test loss and train/test accuracy curves."""
    with open(log_path) as f:
        lines = [json.loads(line) for line in f]
    epoch_records = [line for line in lines if "epoch" in line]

    epochs = [r["epoch"] for r in epoch_records]
    train_loss = [r["train_loss"] for r in epoch_records]
    test_loss = [r["test_loss"] for r in epoch_records]
    train_acc = [r["train_acc"] for r in epoch_records]
    test_acc = [r["test_acc"] for r in epoch_records]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, train_loss, label="train")
    ax_loss.plot(epochs, test_loss, label="test")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, train_acc, label="train")
    ax_acc.plot(epochs, test_acc, label="test")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_confusion_matrix(confusion: torch.Tensor, save_path: str | None = None):
    """confusion: [V,V] int64, confusion[true,pred] = count."""
    confusion_np = confusion.numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(confusion_np, cmap="Blues")
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


@torch.no_grad()
def visualize_predictions(model, dataset, device, num_examples: int = 5, save_path: str | None = None):
    """Shows num_examples test sequences as colored strips (one cell per
    token, colored by token value), each titled with the ground-truth
    majority label vs. the model's predicted label -- the sequence
    analogue of the input/ground-truth/prediction grids used elsewhere
    in this project for images."""
    num_examples = min(num_examples, len(dataset))
    indices = torch.randperm(len(dataset))[:num_examples]

    fig, axes = plt.subplots(num_examples, 1, figsize=(8, 1.1 * num_examples))
    if num_examples == 1:
        axes = [axes]

    for ax, index in zip(axes, indices):
        sequence, label = dataset[index]  # [L], scalar
        tokens = sequence.unsqueeze(0).to(device)  # [1,L]
        logits = model(tokens)["logits"]  # [1,V]
        pred = logits.argmax(dim=-1).item()

        ax.imshow(sequence.unsqueeze(0).numpy(), cmap="tab20", aspect="auto", vmin=0, vmax=model.num_classes - 1)
        for position, token_value in enumerate(sequence.tolist()):
            ax.text(position, 0, str(token_value), ha="center", va="center", fontsize=8)
        ax.set_yticks([])
        ax.set_xticks([])
        correct = "correct" if pred == label.item() else "WRONG"
        ax.set_ylabel(f"true={label.item()} pred={pred}\n({correct})", rotation=0, labelpad=45, fontsize=9, va="center")

    fig.suptitle("Sample test sequences: true vs. predicted majority token")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


@torch.no_grad()
def visualize_cls_attention(model, dataset, device, num_examples: int = 4, layer: int = -1, save_path: str | None = None):
    """For num_examples test sequences, plots the CLS token's attention
    weights (last-layer by default, averaged over heads) to every
    position [CLS, token_0, ..., token_{L-1}], with the actual token
    values as x-axis labels. If the Transformer has learned the task,
    attention mass should concentrate on positions holding the true
    majority value.

    attn_weights[layer] has shape [B,heads,L+1,L+1]; row 0 (query =
    CLS, which is always position 0) gives CLS's attention over all
    key positions, so attn_weights[layer][:, :, 0, :] -> [B,heads,L+1],
    averaged over heads -> [B,L+1].
    """
    num_examples = min(num_examples, len(dataset))
    indices = torch.randperm(len(dataset))[:num_examples]

    fig, axes = plt.subplots(num_examples, 1, figsize=(8, 2.2 * num_examples))
    if num_examples == 1:
        axes = [axes]

    for ax, index in zip(axes, indices):
        sequence, label = dataset[index]  # [L], scalar
        tokens = sequence.unsqueeze(0).to(device)  # [1,L]
        outputs = model(tokens)
        attn = outputs["attn_weights"][layer]  # [1,heads,L+1,L+1]

        cls_attention = attn[0, :, 0, :].mean(dim=0).cpu().numpy()  # [heads,L+1] -> [L+1], averaged over heads
        pred = outputs["logits"].argmax(dim=-1).item()

        position_labels = ["CLS"] + [str(t) for t in sequence.tolist()]
        colors = ["tab:red" if (p > 0 and sequence[p - 1].item() == label.item()) else "tab:gray" for p in range(len(position_labels))]

        ax.bar(range(len(position_labels)), cls_attention, color=colors)
        ax.set_xticks(range(len(position_labels)))
        ax.set_xticklabels(position_labels, fontsize=8)
        ax.set_ylabel("attn. weight")
        ax.set_title(f"true majority={label.item()}, predicted={pred}", fontsize=10)

    fig.suptitle(f"CLS attention, layer {layer}, mean over heads (red = position holds majority value)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def evaluate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_model_and_config(args.checkpoint_path, device)
    _, test_dataset = make_majority_token_datasets(
        config["vocab_size"], config["seq_len"], config["num_train"], config["num_test"], seed=config["seed"]
    )

    os.makedirs(args.output_dir, exist_ok=True)

    accuracy, confusion = compute_test_accuracy_and_confusion(model, test_dataset, device)
    print(f"test accuracy: {accuracy:.4f}  (chance level = {1 / config['vocab_size']:.4f})")

    plot_training_curves(args.log_path, save_path=os.path.join(args.output_dir, "training_curves.png"))
    plot_confusion_matrix(confusion, save_path=os.path.join(args.output_dir, "confusion_matrix.png"))
    visualize_predictions(model, test_dataset, device, num_examples=5, save_path=os.path.join(args.output_dir, "sample_predictions.png"))
    visualize_cls_attention(model, test_dataset, device, num_examples=4, save_path=os.path.join(args.output_dir, "cls_attention.png"))

    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/majority_token/checkpoint.pt")
    parser.add_argument("--log-path", type=str, default="checkpoints/majority_token/train_log.jsonl")
    parser.add_argument("--output-dir", type=str, default="checkpoints/majority_token/figures")
    args = parser.parse_args()
    evaluate(args)
