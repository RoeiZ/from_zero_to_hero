"""Loads a checkpoint saved by vit_train.py and produces the qualitative
+ quantitative evaluation of VisionTransformerClassifier on CIFAR-10's
held-out test split: training curves, overall/per-class test accuracy,
a confusion matrix, sample predictions, and -- the visualization that
is specific to a Transformer operating on image patches -- the
last-layer CLS-to-patch attention, reshaped back into the image's
patch grid and overlaid on the input image.

No training happens here; this script only runs forward passes
(torch.no_grad()) against a checkpoint produced by vit_train.py.

Colab launch example (after vit_train.py has produced a checkpoint +
log on Drive):
    python code/transformers/vit_evaluate.py \
        --checkpoint-path /content/drive/MyDrive/thesis_data/vit_cifar10/checkpoint.pt \
        --log-path /content/drive/MyDrive/thesis_data/vit_cifar10/train_log.jsonl \
        --data-root /content/drive/MyDrive/thesis_data/cifar10 \
        --output-dir /content/drive/MyDrive/thesis_data/vit_cifar10/figures
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from vit_patch_embedding import VisionTransformerClassifier
from vit_train import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, make_cifar10_dataloaders


def load_model_and_config(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = VisionTransformerClassifier(
        config["image_size"], config["patch_size"], config["image_channels"], config["embed_dim"],
        config["num_heads"], config["num_layers"], config["ffn_hidden_dim"], config["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def _unnormalize(images: torch.Tensor) -> torch.Tensor:
    """Inverts the CIFAR10_MEAN/CIFAR10_STD normalization applied in
    vit_train.make_cifar10_dataloaders, for display only.

    Args:
        images: [B,3,H,W] or [3,H,W], normalized
    Returns:
        same shape, clamped to [0,1]
    """
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(-1, 1, 1)  # [3] -> [3,1,1], broadcasts over H,W
    std = torch.tensor(CIFAR10_STD, device=images.device).view(-1, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def compute_test_accuracy_and_confusion(model, test_loader, device):
    """
    Returns:
        accuracy:  float
        confusion: [num_classes,num_classes] int64, confusion[true,pred] = count
    """
    num_classes = model.classifier_head.out_features
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    total_correct = 0
    total_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)["logits"]  # [B,num_classes]
        preds = logits.argmax(dim=-1)  # [B]

        total_correct += (preds == labels).sum().item()
        total_samples += labels.shape[0]
        for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
            confusion[true_label, pred_label] += 1

    accuracy = total_correct / total_samples
    return accuracy, confusion


def plot_training_curves(log_path: str, save_path: str | None = None):
    """Reads the per-epoch JSONL log written by vit_train.py and plots
    train/test loss and train/test accuracy curves."""
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


def plot_confusion_matrix(confusion: torch.Tensor, class_names, save_path: str | None = None):
    """confusion: [num_classes,num_classes] int64, confusion[true,pred] = count."""
    confusion_np = confusion.numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(confusion_np, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


@torch.no_grad()
def visualize_predictions(model, test_dataset, device, class_names, num_examples: int = 5, save_path: str | None = None):
    """Shows num_examples test images with input / true label / predicted
    label together, per the project's results-visualization convention."""
    num_examples = min(num_examples, len(test_dataset))
    indices = torch.randperm(len(test_dataset))[:num_examples]

    fig, axes = plt.subplots(1, num_examples, figsize=(2.6 * num_examples, 3))
    if num_examples == 1:
        axes = [axes]

    for ax, index in zip(axes, indices):
        image, label = test_dataset[index]  # [3,32,32], scalar
        logits = model(image.unsqueeze(0).to(device))["logits"]  # [1,num_classes]
        pred = logits.argmax(dim=-1).item()

        image_display = _unnormalize(image).permute(1, 2, 0).numpy()  # [3,H,W] -> [H,W,3]
        ax.imshow(image_display)
        ax.set_xticks([])
        ax.set_yticks([])
        correct = "correct" if pred == label else "WRONG"
        ax.set_title(f"true: {class_names[label]}\npred: {class_names[pred]}\n({correct})", fontsize=9)

    fig.suptitle("Sample test images: true vs. predicted class")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


@torch.no_grad()
def visualize_cls_attention(model, test_dataset, device, class_names, num_examples: int = 4, layer: int = -1, save_path: str | None = None):
    """For num_examples test images, overlays the CLS token's attention
    to every patch (last layer by default, averaged over heads) on top
    of the input image, reshaped from [N] back into the (H/P, W/P)
    patch grid and nearest-neighbor-upsampled to image resolution.

    attn_weights[layer] has shape [B,heads,N+1,N+1]; row 0 (query = CLS,
    always position 0) gives CLS's attention over all key positions:
    attn_weights[layer][:, :, 0, :] -> [B,heads,N+1]. Column 0 of that
    row is CLS attending to itself and is dropped before reshaping,
    leaving [B,heads,N] -> mean over heads -> [N] -> [H/P,W/P].
    """
    num_examples = min(num_examples, len(test_dataset))
    indices = torch.randperm(len(test_dataset))[:num_examples]
    patch_size = model.patch_size
    num_patches_per_side = model.num_patches_per_side

    fig, axes = plt.subplots(2, num_examples, figsize=(2.6 * num_examples, 5.2))
    if num_examples == 1:
        axes = axes.reshape(2, 1)

    for col, index in enumerate(indices):
        image, label = test_dataset[index]  # [3,32,32], scalar
        outputs = model(image.unsqueeze(0).to(device))
        attn = outputs["attn_weights"][layer]  # [1,heads,N+1,N+1]
        pred = outputs["logits"].argmax(dim=-1).item()

        cls_to_patches = attn[0, :, 0, 1:].mean(dim=0)  # [heads,N+1] -> [heads,N] (drop CLS->CLS) -> [N]
        attn_grid = cls_to_patches.reshape(num_patches_per_side, num_patches_per_side)  # [N] -> [H/P,W/P]
        # Nearest-neighbor upsample [H/P,W/P] -> [H,W] by repeating each
        # patch's scalar attention value over its P x P pixel footprint,
        # so the overlay aligns with the original image resolution.
        attn_upsampled = attn_grid.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)

        image_display = _unnormalize(image).permute(1, 2, 0).cpu().numpy()  # [3,H,W] -> [H,W,3]

        axes[0, col].imshow(image_display)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        correct = "correct" if pred == label else "WRONG"
        axes[0, col].set_title(f"true: {class_names[label]}\npred: {class_names[pred]} ({correct})", fontsize=8)

        axes[1, col].imshow(image_display)
        axes[1, col].imshow(attn_upsampled.cpu().numpy(), cmap="jet", alpha=0.5)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        axes[1, col].set_title("CLS attention", fontsize=8)

    fig.suptitle(f"Last-layer CLS-to-patch attention (layer {layer}, mean over heads)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def evaluate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_model_and_config(args.checkpoint_path, device)
    _, test_loader = make_cifar10_dataloaders(args.data_root, batch_size=256)
    test_dataset = test_loader.dataset

    os.makedirs(args.output_dir, exist_ok=True)

    accuracy, confusion = compute_test_accuracy_and_confusion(model, test_loader, device)
    print(f"test accuracy: {accuracy:.4f}  (chance level = {1 / config['num_classes']:.4f})")

    plot_training_curves(args.log_path, save_path=os.path.join(args.output_dir, "training_curves.png"))
    plot_confusion_matrix(confusion, CIFAR10_CLASSES, save_path=os.path.join(args.output_dir, "confusion_matrix.png"))
    visualize_predictions(
        model, test_dataset, device, CIFAR10_CLASSES, num_examples=5,
        save_path=os.path.join(args.output_dir, "sample_predictions.png"),
    )
    visualize_cls_attention(
        model, test_dataset, device, CIFAR10_CLASSES, num_examples=4,
        save_path=os.path.join(args.output_dir, "cls_attention.png"),
    )

    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/vit_cifar10/checkpoint.pt")
    parser.add_argument("--log-path", type=str, default="checkpoints/vit_cifar10/train_log.jsonl")
    parser.add_argument("--data-root", type=str, default="data/cifar10")
    parser.add_argument("--output-dir", type=str, default="checkpoints/vit_cifar10/figures")
    args = parser.parse_args()
    evaluate(args)
