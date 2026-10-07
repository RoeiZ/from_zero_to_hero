"""Small, fully explicit CNN for CIFAR-10 classification.

Every convolution, ReLU, and pooling operation is a named layer
(no nn.Sequential) and forward() prints the [B, C, H, W] tensor shape
after each individual operation when verbose=True. This is a teaching
exercise: the goal is to make shape bookkeeping automatic before
moving on to architectures (U-Net, ViT, DiT) where it gets harder.

Local usage (lightweight smoke test only, no dataset download):
    python code/cifar10_explicit_cnn.py --smoke-test

Colab usage (real data, full training):
    python code/cifar10_explicit_cnn.py --data-root /content/data \
        --epochs 10 --batch-size 128 --lr 1e-3
"""

import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


class ExplicitCIFAR10CNN(nn.Module):
    """Three conv blocks + two FC layers, every op named explicitly.
why 
    Shape trace for input [B, 3, 32, 32]:
        conv1 (3->16, k=3, pad=1) -> [B, 16, 32, 32]
        relu1                     -> [B, 16, 32, 32]
        pool1 (2x2)               -> [B, 16, 16, 16]
        conv2 (16->32, k=3, pad=1)-> [B, 32, 16, 16]
        relu2                     -> [B, 32, 16, 16]
        pool2 (2x2)               -> [B, 32, 8, 8]
        conv3 (32->64, k=3, pad=1)-> [B, 64, 8, 8]
        relu3                     -> [B, 64, 8, 8]
        pool3 (2x2)               -> [B, 64, 4, 4]
        flatten                   -> [B, 1024]
        fc1 (1024->128)           -> [B, 128]
        relu4                     -> [B, 128]
        fc2 (128->10)             -> [B, 10]   (logits)

    Padding=1 with a 3x3 kernel keeps H,W unchanged through each conv
    (out = in + 2*pad - kernel + 1 = in), so only the 2x2 max-pool
    layers (stride 2) halve the spatial resolution: 32 -> 16 -> 8 -> 4.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- conv block 1 ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # --- conv block 2 ---
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # --- conv block 3 ---
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # --- classifier head ---
        self.flatten = nn.Flatten()  # [B, 64, 4, 4] -> [B, 64*4*4] = [B, 1024]
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """x: [B, 3, 32, 32] -> logits: [B, num_classes]."""

        def step(layer, h, name):
            h = layer(h)
            if verbose:
                print(f"after {name:<8}: {tuple(h.shape)}")
            return h

        if verbose:
            print(f"input          : {tuple(x.shape)}")

        h = step(self.conv1, x, "conv1")
        h = step(self.relu1, h, "relu1")
        h = step(self.pool1, h, "pool1")

        h = step(self.conv2, h, "conv2")
        h = step(self.relu2, h, "relu2")
        h = step(self.pool2, h, "pool2")

        h = step(self.conv3, h, "conv3")
        h = step(self.relu3, h, "relu3")
        h = step(self.pool3, h, "pool3")

        h = step(self.flatten, h, "flatten")
        h = step(self.fc1, h, "fc1")
        h = step(self.relu4, h, "relu4")
        logits = step(self.fc2, h, "fc2")

        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_cifar10_dataloaders(data_root: str, batch_size: int):
    """Real CIFAR-10 train/test split. Downloads ~170MB on first call.

    Intended for Colab, not for local smoke tests.
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.ToTensor()  # [0,255] uint8 HWC -> [0,1] float32 CHW

    cifar10_train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform,
    )
    cifar10_test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(cifar10_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def visualize_first_conv_feature_maps(
    model: ExplicitCIFAR10CNN,
    image: torch.Tensor,
    num_maps: int = 10,
    save_path: str | None = None,
):
    """Plot the first `num_maps` feature maps produced by conv1.

    image: [3, 32, 32] or [1, 3, 32, 32], single sample, already on the
    same device as `model`.

    These are the *raw conv1 output* (pre-ReLU), i.e. the literal
    "feature maps of the first convolution": conv1(image) has shape
    [1, 16, 32, 32]; we take the first `num_maps` of the 16 output
    channels and plot each as a grayscale image.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [3,32,32] -> [1,3,32,32]

    model.eval()
    with torch.no_grad():
        feature_maps = model.conv1(image)  # [1, 16, 32, 32]
    feature_maps = feature_maps[0, :num_maps].cpu()  # [num_maps, 32, 32]

    n_cols = 5
    n_rows = (num_maps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    axes = axes.flatten()

    for i in range(num_maps):
        axes[i].imshow(feature_maps[i], cmap="viridis")
        axes[i].set_title(f"channel {i}", fontsize=9)
        axes[i].axis("off")
    for i in range(num_maps, len(axes)):
        axes[i].axis("off")

    fig.suptitle("conv1 feature maps (first 10 of 16 output channels)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved feature-map figure to {save_path}")
    else:
        plt.show()
    input("Continue")
    plt.close(fig)


def visualize_predictions(
    model: ExplicitCIFAR10CNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: tuple = CIFAR10_CLASSES,
    num_examples: int = 5,
    save_path: str | None = None,
):
    """Qualitative check: show `num_examples` images next to their
    ground-truth label and the model's predicted label (green title if
    correct, red if wrong).

    images: [N, 3, 32, 32] float in [0,1] (ToTensor output), already on
    the same device as `model`. labels: [N] int64 ground-truth class
    indices, any device.
    """
    num_examples = min(num_examples, images.size(0))

    model.eval()
    with torch.no_grad():
        logits = model(images[:num_examples])  # [num_examples, 10]
    predicted = logits.argmax(dim=1).cpu()  # [num_examples]
    images_cpu = images[:num_examples].cpu()  # [num_examples, 3, 32, 32]
    labels_cpu = labels[:num_examples].cpu()  # [num_examples]

    fig, axes = plt.subplots(1, num_examples, figsize=(2.6 * num_examples, 3.0))
    if num_examples == 1:
        axes = [axes]

    for i in range(num_examples):
        image_hwc = images_cpu[i].permute(1, 2, 0).numpy()  # [3,32,32] -> [32,32,3] for imshow
        axes[i].imshow(image_hwc)
        true_name = class_names[labels_cpu[i].item()]
        pred_name = class_names[predicted[i].item()]
        is_correct = predicted[i].item() == labels_cpu[i].item()
        axes[i].set_title(
            f"true: {true_name}\npred: {pred_name}",
            color=("green" if is_correct else "red"),
            fontsize=9,
        )
        axes[i].axis("off")

    fig.suptitle(f"sample predictions ({num_examples} test examples)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved predictions figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)  # [B,3,32,32], [B]

        optimizer.zero_grad()
        logits = model(images)          # [B, 10]
        loss = loss_fn(logits, labels)  # scalar
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, no real training. Confirms construction, the
    full forward shape trace, parameter count, one backward pass, one
    optimizer step, and the feature-map plotting code path.
    """
    torch.manual_seed(0)

    model = ExplicitCIFAR10CNN(num_classes=10).to(device)
    n_params = count_trainable_parameters(model)
    print(f"trainable parameters: {n_params:,}")

    dummy_images = torch.randn(4, 3, 32, 32, device=device)  # [B,3,32,32]
    dummy_labels = torch.randint(0, 10, (4,), device=device)  # [B]

    print("\n--- forward shape trace ---")
    logits = model(dummy_images, verbose=True)
    assert logits.shape == (4, 10), f"unexpected logits shape {tuple(logits.shape)}"

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = loss_fn(logits, dummy_labels)
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    optimizer.step()
    print(f"\ndummy loss: {loss.item():.4f} (finite, backward + optimizer step ok)")

    scratch_png = "cifar10_explicit_cnn_smoke_feature_maps.png"
    visualize_first_conv_feature_maps(
        model, dummy_images[0], num_maps=10, save_path=scratch_png,
    )

    scratch_predictions_png = "cifar10_explicit_cnn_smoke_predictions.png"
    visualize_predictions(
        model, dummy_images, dummy_labels, num_examples=4, save_path=scratch_predictions_png,
    )

    print("\nsmoke test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="run lightweight local verification with synthetic tensors, no download")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    torch.manual_seed(args.seed)

    model = ExplicitCIFAR10CNN(num_classes=10).to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    train_loader, test_loader = get_cifar10_dataloaders(args.data_root, args.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)
    visualize_first_conv_feature_maps(
        model, sample_images[0], num_maps=10,
        save_path="cifar10_conv1_feature_maps.png",
    )
    visualize_predictions(
        model, sample_images, sample_labels, num_examples=5,
        save_path="cifar10_sample_predictions.png",
    )


if __name__ == "__main__":
    main()
