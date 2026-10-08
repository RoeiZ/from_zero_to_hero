"""Compact convolutional autoencoder for MNIST (28x28x1 grayscale digits).

Every encoder/decoder operation is a named layer (no nn.Sequential) and
forward() prints the [B, C, H, W] (or [B, D] in the bottleneck) tensor
shape after each individual operation when verbose=True.

The network minimizes reconstruction error:

    L(x) = || x - decoder(encoder(x)) ||^2        (pixel-wise MSE)

x is the input image itself (no label is used): the encoder compresses
x into a low-dimensional latent vector z = encoder(x) in R^latent_dim,
and the decoder reconstructs x_hat = decoder(z) from z alone. This is
exact mean-squared-error reconstruction loss, not an approximation.

Local usage (lightweight smoke test only, no dataset download):
    python code/mnist_conv_autoencoder.py --smoke-test

Colab usage (real data, full training):
    python code/mnist_conv_autoencoder.py --data-root /content/data \
        --epochs 10 --batch-size 128 --lr 1e-3
"""

import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ConvAutoencoder(nn.Module):
    """3 strided conv blocks (encoder) + FC bottleneck + 3 transposed-conv
    blocks (decoder), every op named explicitly.

    Shape trace for input [B, 1, 28, 28] with latent_dim=32:

        --- encoder ---
        conv1  (1 ->16, k3, s2, p1) -> [B, 16, 14, 14]
        relu1                        -> [B, 16, 14, 14]
        conv2  (16->32, k3, s2, p1) -> [B, 32,  7,  7]
        relu2                        -> [B, 32,  7,  7]
        conv3  (32->64, k3, s2, p1) -> [B, 64,  4,  4]
        relu3                        -> [B, 64,  4,  4]
        flatten                      -> [B, 1024]
        fc_enc (1024->32)            -> [B,  32]           (latent z)

        --- decoder ---
        fc_dec  (32->1024)           -> [B, 1024]
        relu_dec                     -> [B, 1024]
        unflatten                    -> [B, 64, 4, 4]
        deconv1 (64->32, k3,s2,p1,op0) -> [B, 32,  7,  7]
        relu_d1                        -> [B, 32,  7,  7]
        deconv2 (32->16, k3,s2,p1,op1) -> [B, 16, 14, 14]
        relu_d2                        -> [B, 16, 14, 14]
        deconv3 (16-> 1, k3,s2,p1,op1) -> [B,  1, 28, 28]
        sigmoid                        -> [B,  1, 28, 28]   (x_hat in [0,1])

    Stride-2 3x3 convolutions with padding=1 halve H,W at each encoder
    stage (28->14->7->4), following
        out = floor((in + 2*pad - kernel) / stride) + 1.
    The decoder undoes this with ConvTranspose2d, which follows
        out = (in - 1)*stride - 2*pad + kernel + output_padding.
    `output_padding` (op) is needed because stride-2 downsampling is not
    exactly invertible: 7 -> 4 -> 7 loses no information about the
    *formula*, but 4 -> 7 alone is ambiguous between output sizes 7 and
    8 (both satisfy the floor()); output_padding picks 7. Concretely:
      deconv1: in=4  -> (4-1)*2-2*1+3+0 = 7   (matches encoder's 7)
      deconv2: in=7  -> (7-1)*2-2*1+3+1 = 14  (matches encoder's 14)
      deconv3: in=14 -> (14-1)*2-2*1+3+1 = 28 (matches encoder's 28, the input)
    A final sigmoid squashes the output to [0,1], matching the range of
    ToTensor()-normalized MNIST pixels, so the reconstruction loss
    compares values on the same scale as the input.
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        # --- encoder: 28x28 -> 14x14 -> 7x7 -> 4x4, channels 1->16->32->64 ---
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()

        # --- bottleneck: [B,64,4,4] <-> [B,1024] <-> [B,latent_dim] ---
        self.flatten = nn.Flatten()  # [B,64,4,4] -> [B, 64*4*4] = [B,1024]
        self.fc_enc = nn.Linear(in_features=64 * 4 * 4, out_features=latent_dim)
        self.fc_dec = nn.Linear(in_features=latent_dim, out_features=64 * 4 * 4)
        self.relu_dec = nn.ReLU()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 4, 4))  # [B,1024] -> [B,64,4,4]

        # --- decoder: 4x4 -> 7x7 -> 14x14 -> 28x28, channels 64->32->16->1 ---
        # output_padding breaks the stride-2 output-size ambiguity so the
        # decoder exactly mirrors the encoder's spatial sizes (see class docstring).
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.relu_d1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_d2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """x: [B, 1, 28, 28] in [0,1] -> x_hat: [B, 1, 28, 28] in [0,1]."""

        def step(layer, h, name):
            h = layer(h)
            if verbose:
                print(f"after {name:<9}: {tuple(h.shape)}")
            return h

        if verbose:
            print(f"input          : {tuple(x.shape)}")

        h = step(self.conv1, x, "conv1")
        h = step(self.relu1, h, "relu1")
        h = step(self.conv2, h, "conv2")
        h = step(self.relu2, h, "relu2")
        h = step(self.conv3, h, "conv3")
        h = step(self.relu3, h, "relu3")

        h = step(self.flatten, h, "flatten")
        z = step(self.fc_enc, h, "fc_enc")  # [B, latent_dim], the bottleneck

        h = step(self.fc_dec, z, "fc_dec")
        h = step(self.relu_dec, h, "relu_dec")
        h = step(self.unflatten, h, "unflatten")

        h = step(self.deconv1, h, "deconv1")
        h = step(self.relu_d1, h, "relu_d1")
        h = step(self.deconv2, h, "deconv2")
        h = step(self.relu_d2, h, "relu_d2")
        h = step(self.deconv3, h, "deconv3")
        x_hat = step(self.sigmoid, h, "sigmoid")

        return x_hat


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_mnist_dataloaders(data_root: str, batch_size: int):
    """Real MNIST train/test split (torchvision's native partition).
    Downloads ~12MB on first call. Intended for Colab, not local smoke tests.
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.ToTensor()  # [0,255] uint8 HxW -> [0,1] float32 [1,28,28]

    mnist_train_dataset = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform,
    )
    mnist_test_dataset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def visualize_reconstructions(
    model: ConvAutoencoder,
    images: torch.Tensor,
    labels: torch.Tensor | None = None,
    num_examples: int = 5,
    save_path: str | None = None,
):
    """Qualitative + quantitative check: top row = original images,
    bottom row = the model's reconstruction, with the per-image
    reconstruction MSE (computed against that original image) in each
    bottom title.

    images: [N, 1, 28, 28] float in [0,1] (ToTensor output), already on
    the same device as `model`. labels: optional [N] int64 digit labels
    (any device), shown above the originals purely for readability.
    """
    num_examples = min(num_examples, images.size(0))

    model.eval()
    with torch.no_grad():
        originals = images[:num_examples]            # [num_examples, 1, 28, 28]
        reconstructions = model(originals)            # [num_examples, 1, 28, 28]
        per_image_mse = ((reconstructions - originals) ** 2).mean(dim=(1, 2, 3))  # [num_examples]

    originals_cpu = originals.cpu()
    reconstructions_cpu = reconstructions.cpu()
    per_image_mse_cpu = per_image_mse.cpu()
    labels_cpu = labels[:num_examples].cpu() if labels is not None else None

    fig, axes = plt.subplots(2, num_examples, figsize=(2.0 * num_examples, 4.2))
    if num_examples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_examples):
        # vmin/vmax=0,1 pin both rows to the same grayscale scale (the
        # true pixel range after ToTensor) so a dim reconstruction reads
        # as genuinely dim, not auto-contrast-stretched to look sharper.
        axes[0, i].imshow(originals_cpu[i, 0], cmap="gray", vmin=0, vmax=1)
        top_title = f"true: {labels_cpu[i].item()}" if labels_cpu is not None else "original"
        axes[0, i].set_title(top_title, fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructions_cpu[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"MSE={per_image_mse_cpu[i].item():.4f}", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle(f"original (top) vs. reconstruction (bottom), {num_examples} examples")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved reconstruction figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_samples = 0.0, 0

    for images, _labels in loader:
        images = images.to(device)  # [B,1,28,28]; label unused, AE target is the image itself

        optimizer.zero_grad()
        reconstructions = model(images)       # [B,1,28,28]
        loss = loss_fn(reconstructions, images)  # scalar MSE
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_samples = 0.0, 0

    for images, _labels in loader:
        images = images.to(device)
        reconstructions = model(images)
        loss = loss_fn(reconstructions, images)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, no real training. Confirms construction, the
    full forward shape trace, parameter count, one backward pass, one
    optimizer step, and the reconstruction-plotting code path.
    """
    torch.manual_seed(0)

    model = ConvAutoencoder(latent_dim=32).to(device)
    n_params = count_trainable_parameters(model)
    print(f"trainable parameters: {n_params:,}")

    dummy_images = torch.rand(4, 1, 28, 28, device=device)  # [B,1,28,28], in [0,1] like real MNIST
    dummy_labels = torch.randint(0, 10, (4,), device=device)  # [B]

    print("\n--- forward shape trace ---")
    reconstructions = model(dummy_images, verbose=True)
    assert reconstructions.shape == dummy_images.shape, (
        f"reconstruction shape {tuple(reconstructions.shape)} != input shape {tuple(dummy_images.shape)}"
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = loss_fn(reconstructions, dummy_images)
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    optimizer.step()
    print(f"\ndummy reconstruction MSE: {loss.item():.4f} (finite, backward + optimizer step ok)")

    scratch_png = "mnist_autoencoder_smoke_reconstructions.png"
    visualize_reconstructions(
        model, dummy_images, dummy_labels, num_examples=4, save_path=scratch_png,
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
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    torch.manual_seed(args.seed)

    model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    train_loader, test_loader = get_mnist_dataloaders(args.data_root, args.batch_size)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss = evaluate(model, test_loader, loss_fn, device)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_mse={train_loss:.4f}  test_mse={test_loss:.4f}")

    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)
    visualize_reconstructions(
        model, sample_images, sample_labels, num_examples=5,
        save_path="mnist_autoencoder_reconstructions.png",
    )


if __name__ == "__main__":
    main()
