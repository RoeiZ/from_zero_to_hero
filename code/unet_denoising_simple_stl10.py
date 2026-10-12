"""Simple, single-objective U-Net denoiser -- STL-10, native 96x96.

Same simple setup as code/unet_denoising_simple.py (CIFAR-10 @ 64x64,
single "x0" objective, no comparison script, no prediction_mode
bookkeeping), just moved to a higher-resolution dataset: STL-10's
native 96x96x3 images (no resizing/cropping), using the SmallUNet
architecture already defined in code/stl10_unet.py (3 encoder/decoder
levels, 96->48->24->12, convolutional bottleneck, no FC layer) rather
than TinyUNet64 (which was built for 64x64 and would need 96x96 inputs
either resized down or padded -- reusing SmallUNet avoids that
entirely, since it was already sized for 96x96).

SmallUNet was originally built in code/stl10_unet.py for PLAIN
reconstruction (no corruption): L = ||x - SmallUNet(x)||^2. Reused here
for DENOISING by corrupting the input first:

    x_noisy = x + sigma*eps,  eps ~ N(0,I),  sigma = noise_std
    L = ||x - SmallUNet(x_noisy)||^2

No change to the model itself is needed for this: SmallUNet already
ends in a Sigmoid and outputs x_hat directly (the "x0" objective in
code/unet_tiny_denoising_64.py's terminology), so only the
train/eval loop differs from stl10_unet.py's own (it corrupts the
input before the forward pass; the loss target is still the clean
image).

Because there is only one objective here, the training loss and the
test-time denoising-quality metric are the exact same formula --
MSE(x_hat, x) -- computed on different data splits; see
code/unet_denoising_simple.py's module docstring for the fuller
train-vs-test explanation (identical reasoning applies here).

Local usage (lightweight smoke test only, no dataset download):
    python code/unet_denoising_simple_stl10.py --smoke-test

Colab usage (real data, full training). STL-10 is a ~2.6GB download, so
this must run on Colab with Drive caching, not locally:
    python code/unet_denoising_simple_stl10.py --data-root /content/data \
        --epochs 15 --batch-size 64 --lr 1e-3 --noise-std 0.3
"""

import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from stl10_unet import SmallUNet, count_trainable_parameters, psnr_from_mse, get_stl10_dataloaders
from unet_tiny_denoising_64 import add_gaussian_noise


def train_one_epoch(model, loader, loss_fn, optimizer, device, noise_std):
    """One training epoch: corrupt each clean image, run SmallUNet,
    compute MSE(x_hat, clean), backprop, step. Returns the average
    TRAIN loss -- computed while parameters are still being updated
    (model.train(), gradients on), so it is a convergence signal, not a
    reliable performance estimate (see `evaluate`)."""
    model.train()
    total_loss, total_samples = 0.0, 0

    for clean_images, _labels in loader:
        clean_images = clean_images.to(device)  # [B,3,96,96]; label unused
        noisy_images = add_gaussian_noise(clean_images, noise_std)

        optimizer.zero_grad()
        x_hat = model(noisy_images)              # [B,3,96,96], in [0,1]
        loss = loss_fn(x_hat, clean_images)       # scalar MSE
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clean_images.size(0)
        total_samples += clean_images.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, noise_std):
    """Average TEST-set denoising MSE: same formula as
    `train_one_epoch`, but with the model frozen (model.eval(),
    no_grad) on held-out images -- the number that actually answers
    "how well does this U-Net denoise STL-10 images"."""
    model.eval()
    total_loss, total_samples = 0.0, 0

    for clean_images, _labels in loader:
        clean_images = clean_images.to(device)
        noisy_images = add_gaussian_noise(clean_images, noise_std)

        x_hat = model(noisy_images)
        loss = loss_fn(x_hat, clean_images)

        total_loss += loss.item() * clean_images.size(0)
        total_samples += clean_images.size(0)

    return total_loss / total_samples


def visualize_denoising(
    model: SmallUNet,
    clean_images: torch.Tensor,
    noise_std: float,
    num_examples: int = 5,
    save_path: str | None = None,
):
    """Qualitative check: row 1 = clean (ground truth), row 2 = noisy
    input, row 3 = SmallUNet's x_hat, with per-image MSE/PSNR (output
    vs. clean) in each row-3 title. clean_images: [N,3,96,96] float in
    [0,1], already on model's device."""
    num_examples = min(num_examples, clean_images.size(0))

    model.eval()
    with torch.no_grad():
        clean = clean_images[:num_examples]                 # [num_examples,3,96,96]
        noisy = add_gaussian_noise(clean, noise_std)         # [num_examples,3,96,96]
        denoised = model(noisy)                              # [num_examples,3,96,96]
        per_image_mse = ((denoised - clean) ** 2).mean(dim=(1, 2, 3))  # [num_examples]
        per_image_psnr = psnr_from_mse(per_image_mse)          # [num_examples], dB

    clean_cpu, noisy_cpu, denoised_cpu = clean.cpu(), noisy.cpu(), denoised.cpu()
    mse_cpu, psnr_cpu = per_image_mse.cpu(), per_image_psnr.cpu()

    def to_hwc(img_chw):
        return img_chw.permute(1, 2, 0).numpy()  # [3,96,96] -> [96,96,3]

    fig, axes = plt.subplots(3, num_examples, figsize=(2.2 * num_examples, 6.6))
    if num_examples == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["clean (target)", "noisy (input)", "SmallUNet x_hat"]
    for i in range(num_examples):
        axes[0, i].imshow(to_hwc(clean_cpu[i]))
        axes[0, i].set_title(row_titles[0] if i == 0 else "", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(to_hwc(noisy_cpu[i]))
        axes[1, i].set_title(row_titles[1] if i == 0 else "", fontsize=9)
        axes[1, i].axis("off")

        axes[2, i].imshow(to_hwc(denoised_cpu[i]))
        axes[2, i].set_title(f"MSE={mse_cpu[i].item():.4f}\nPSNR={psnr_cpu[i].item():.1f}dB", fontsize=8)
        axes[2, i].axis("off")

    fig.suptitle(f"clean / noisy (std={noise_std}) / SmallUNet-denoised, {num_examples} examples")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved denoising figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, 2 tiny epochs. Confirms construction, one
    training epoch, one evaluation pass, finite loss, and the
    denoising-visualization code path at STL-10's native 96x96
    resolution."""
    torch.manual_seed(0)
    from torch.utils.data import DataLoader, TensorDataset

    model = SmallUNet().to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    synthetic_images = torch.rand(8, 3, 96, 96, device=device)  # [8,3,96,96], in [0,1]
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

    scratch_png = "unet_denoising_simple_stl10_smoke.png"
    visualize_denoising(model, synthetic_images, noise_std, num_examples=4, save_path=scratch_png)

    print("\nsmoke test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="run lightweight local verification with synthetic tensors, no download")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.3,
                         help="std of the additive Gaussian noise used to create the denoising task")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    torch.manual_seed(args.seed)

    model = SmallUNet().to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    train_loader, test_loader = get_stl10_dataloaders(args.data_root, args.batch_size)

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
        save_path="unet_denoising_simple_stl10_reconstructions.png",
    )


if __name__ == "__main__":
    main()
