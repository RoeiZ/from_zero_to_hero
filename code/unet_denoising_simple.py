"""Simple, single-objective U-Net denoiser.

Trains the TinyUNet64 architecture (defined in
code/unet_tiny_denoising_64.py) with the standard "x0" objective --
the network directly predicts the clean image, x_hat = UNet(x_noisy)
-- and reports its denoising performance. No second model, no
objective comparison, no prediction_mode bookkeeping: this script
exists purely to look at "how well does this U-Net denoise?" without
the x0-vs-noise-objective machinery in
code/compare_denoising_objectives.py (which trains two models and
needs to distinguish each objective's native loss units from the
common, comparable metric -- not needed here, since there is only one
objective).

Task: a clean image x is corrupted with additive Gaussian noise to
make x_noisy = x + sigma*eps, eps ~ N(0,I), sigma = noise_std, and the
U-Net is trained to map x_noisy -> x_hat ~= x via
L = ||x - UNet(x_noisy)||^2 (MSE, identical formula for both training
and evaluation). Because the model's raw output already IS the
clean-image estimate in this objective, the training loss and the
test-time denoising-quality metric are the exact same formula --
MSE(x_hat, x) -- just computed on different data splits (see the
printed epoch line, and the explanation in
`train_one_epoch`/`evaluate`'s docstrings below).

Local usage (lightweight smoke test only, no dataset download):
    python code/unet_denoising_simple.py --smoke-test

Colab usage (real data, full training). Uses CIFAR-10 (small download,
~170MB) upsampled to 64x64 so the U-Net operates at the resolution it
was designed for:
    python code/unet_denoising_simple.py --data-root /content/data \
        --epochs 10 --batch-size 128 --lr 1e-3 --noise-std 0.3
"""

import argparse

import torch
import torch.nn as nn

from unet_tiny_denoising_64 import (
    TinyUNet64,
    count_trainable_parameters,
    get_cifar10_denoising_dataloaders,
    visualize_denoising,
)


def train_one_epoch(model, loader, loss_fn, optimizer, device, noise_std):
    """One pass over the training set. For each batch: corrupt the
    clean image, run the U-Net, compute MSE(x_hat, clean), backprop,
    step. Returns the average TRAIN loss -- computed on batches the
    optimizer is actively learning from (model.train(), gradients
    enabled, parameters change mid-epoch), so it reflects a mix of
    "older" and "just-updated" parameters rather than one fixed
    snapshot. Mainly useful to watch for convergence/divergence
    during training, not as a reliable performance estimate (a model
    can fit/memorize its own training batches better than it
    generalizes)."""
    model.train()
    total_loss, total_samples = 0.0, 0

    for clean_images, _labels in loader:
        clean_images = clean_images.to(device)  # [B,3,64,64]; label unused
        noisy_images = clean_images + noise_std * torch.randn_like(clean_images)
        noisy_images = noisy_images.clamp(0.0, 1.0)

        optimizer.zero_grad()
        x_hat = model(noisy_images)              # [B,3,64,64], in [0,1] (x0 objective -> Sigmoid output)
        loss = loss_fn(x_hat, clean_images)       # scalar MSE
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clean_images.size(0)
        total_samples += clean_images.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, noise_std):
    """Average TEST-set denoising MSE: the SAME formula as
    `train_one_epoch` (MSE(x_hat, clean)), but computed with the model
    frozen (model.eval(), no gradients, no optimizer step) on held-out
    images the model never trained on. This is the number that
    actually answers "how well does the U-Net denoise", since the
    train loss above can look good even when the model has merely
    memorized training-batch noise patterns rather than learned a
    generalizable denoising function."""
    model.eval()
    total_loss, total_samples = 0.0, 0

    for clean_images, _labels in loader:
        clean_images = clean_images.to(device)
        noisy_images = clean_images + noise_std * torch.randn_like(clean_images)
        noisy_images = noisy_images.clamp(0.0, 1.0)

        x_hat = model(noisy_images)
        loss = loss_fn(x_hat, clean_images)

        total_loss += loss.item() * clean_images.size(0)
        total_samples += clean_images.size(0)

    return total_loss / total_samples


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, 2 tiny epochs. Confirms construction, one
    training epoch, one evaluation pass, finite loss, and the
    denoising-visualization code path."""
    torch.manual_seed(0)
    from torch.utils.data import DataLoader, TensorDataset

    model = TinyUNet64(in_channels=3, out_channels=3, base_channels=32, prediction_mode="x0").to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    synthetic_images = torch.rand(8, 3, 64, 64, device=device)  # [8,3,64,64], in [0,1]
    synthetic_labels = torch.zeros(8, dtype=torch.long, device=device)  # unused, placeholder
    synthetic_loader = DataLoader(TensorDataset(synthetic_images, synthetic_labels), batch_size=4)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    noise_std = 0.3

    for epoch in range(2):
        train_loss = train_one_epoch(model, synthetic_loader, loss_fn, optimizer, device, noise_std)
        test_mse = evaluate(model, synthetic_loader, loss_fn, device, noise_std)
        print(f"epoch {epoch + 1}/2  train_loss={train_loss:.4f}  test_denoising_mse={test_mse:.4f}")
    assert torch.isfinite(torch.tensor(test_mse)), "test_denoising_mse is not finite"

    scratch_png = "unet_denoising_simple_smoke.png"
    visualize_denoising(model, synthetic_images, noise_std, num_examples=4, save_path=scratch_png)

    print("\nsmoke test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="run lightweight local verification with synthetic tensors, no download")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.3,
                         help="std of the additive Gaussian noise used to create the denoising task")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    torch.manual_seed(args.seed)

    model = TinyUNet64(
        in_channels=3, out_channels=3, base_channels=args.base_channels, prediction_mode="x0",
    ).to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    train_loader, test_loader = get_cifar10_denoising_dataloaders(args.data_root, args.batch_size, image_size=64)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.noise_std)
        test_mse = evaluate(model, test_loader, loss_fn, device, args.noise_std)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_loss={train_loss:.4f}  test_denoising_mse={test_mse:.4f}")

    sample_images, _sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)
    visualize_denoising(
        model, sample_images, args.noise_std, num_examples=5,
        save_path="unet_denoising_simple_reconstructions.png",
    )


if __name__ == "__main__":
    main()
