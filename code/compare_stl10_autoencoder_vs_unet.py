"""Reconstruction comparison: plain conv autoencoder vs. small U-Net,
both on STL-10 (96x96x3 RGB), both trained with identical
hyperparameters (same seed, epochs, batch size, learning rate, same
train/test split, same MSE loss, no noise added -- plain reconstruction
L(x) = ||x - model(x)||^2) so any gap in reconstruction quality reflects
the architecture difference, not the training setup.

Models compared (both already defined elsewhere in this repo, imported
here rather than redefined):
    code/stl10_conv_autoencoder.py :: STL10ConvAutoencoder
        4 strided-conv encoder stages -> FC bottleneck (latent_dim=128,
        a flattened vector) -> 4 transposed-conv decoder stages. Every
        pixel must be reconstructed from those 128 numbers alone.
    code/stl10_unet.py :: SmallUNet
        3 strided-conv encoder stages -> convolutional bottleneck (a
        12x12x256 feature map, no FC layer) -> 3 transposed-conv
        decoder stages, each one concatenated with its matching
        encoder skip tensor. Fine spatial detail can bypass the
        bottleneck entirely via the skip connections.

What is controlled: random seed (set immediately before each model's
construction and before each model's training loop), epochs, batch
size, learning rate, optimizer (Adam), loss function (MSELoss), and
the dataset/train-test split (the same two DataLoaders are reused for
both models). What is NOT bit-identical: the two models consume the
shared DataLoader's shuffling RNG independently once their own training
loop starts, so the exact batch order differs between the autoencoder
run and the U-Net run -- standard practice for this kind of comparison,
but worth stating explicitly (see CLAUDE.md Research Integrity).

Local usage (lightweight smoke test only, no dataset download):
    python code/compare_stl10_autoencoder_vs_unet.py --smoke-test

Colab usage (real data, full training of both models):
    python code/compare_stl10_autoencoder_vs_unet.py --data-root /content/data \
        --epochs 15 --batch-size 64 --lr 1e-3
"""

import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from stl10_conv_autoencoder import (
    STL10ConvAutoencoder,
    get_stl10_dataloaders,
    count_trainable_parameters,
    psnr_from_mse,
    ssim,
    train_one_epoch as train_autoencoder_one_epoch,
    evaluate as evaluate_autoencoder,
)
from stl10_unet import (
    SmallUNet,
    train_one_epoch as train_unet_one_epoch,
    evaluate as evaluate_unet,
    evaluate_ssim as evaluate_unet_ssim,
)


@torch.no_grad()
def evaluate_autoencoder_ssim(model, loader, device, window_size: int = 11):
    """Average SSIM over an entire loader for STL10ConvAutoencoder.
    Mirrors stl10_unet.py::evaluate_ssim; not defined in
    stl10_conv_autoencoder.py, so it lives here instead."""
    model.eval()
    total_ssim, total_samples = 0.0, 0

    for images, _labels in loader:
        images = images.to(device)
        reconstructions = model(images)
        batch_ssim = ssim(reconstructions, images, window_size=window_size)  # [B]

        total_ssim += batch_ssim.sum().item()
        total_samples += images.size(0)

    return total_ssim / total_samples


def run_comparison(
    data_root: str,
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-3,
    latent_dim: int = 128,
    device: torch.device | None = None,
    seed: int = 0,
):
    """Train STL10ConvAutoencoder and SmallUNet from scratch, same
    hyperparameters, same train/test loaders. Returns a dict with both
    trained models, both loss histories, both final test metrics
    (mse, psnr, ssim), and the test_loader (for later visualization)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_stl10_dataloaders(data_root, batch_size)
    loss_fn = nn.MSELoss()

    # --- train the plain autoencoder ---
    torch.manual_seed(seed)
    autoencoder = STL10ConvAutoencoder(latent_dim=latent_dim).to(device)
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    autoencoder_history = {"train_mse": [], "test_mse": []}

    for epoch in range(epochs):
        train_mse = train_autoencoder_one_epoch(autoencoder, train_loader, loss_fn, autoencoder_optimizer, device)
        test_mse = evaluate_autoencoder(autoencoder, test_loader, loss_fn, device)
        autoencoder_history["train_mse"].append(train_mse)
        autoencoder_history["test_mse"].append(test_mse)
        print(f"[autoencoder] epoch {epoch + 1:3d}/{epochs}  train_mse={train_mse:.4f}  test_mse={test_mse:.4f}")

    # --- train the U-Net ---
    torch.manual_seed(seed)
    unet = SmallUNet().to(device)
    unet_optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    unet_history = {"train_mse": [], "test_mse": []}

    for epoch in range(epochs):
        train_mse = train_unet_one_epoch(unet, train_loader, loss_fn, unet_optimizer, device)
        test_mse = evaluate_unet(unet, test_loader, loss_fn, device)
        unet_history["train_mse"].append(train_mse)
        unet_history["test_mse"].append(test_mse)
        print(f"[unet]        epoch {epoch + 1:3d}/{epochs}  train_mse={train_mse:.4f}  test_mse={test_mse:.4f}")

    # --- final test-set metrics, both models ---
    autoencoder_test_mse = evaluate_autoencoder(autoencoder, test_loader, loss_fn, device)
    unet_test_mse = evaluate_unet(unet, test_loader, loss_fn, device)
    autoencoder_test_ssim = evaluate_autoencoder_ssim(autoencoder, test_loader, device)
    unet_test_ssim = evaluate_unet_ssim(unet, test_loader, device)

    summary = {
        "autoencoder": {
            "n_params": count_trainable_parameters(autoencoder),
            "test_mse": autoencoder_test_mse,
            "test_psnr_db": psnr_from_mse(torch.tensor(autoencoder_test_mse)).item(),
            "test_ssim": autoencoder_test_ssim,
        },
        "unet": {
            "n_params": count_trainable_parameters(unet),
            "test_mse": unet_test_mse,
            "test_psnr_db": psnr_from_mse(torch.tensor(unet_test_mse)).item(),
            "test_ssim": unet_test_ssim,
        },
    }

    return {
        "autoencoder": autoencoder,
        "unet": unet,
        "autoencoder_history": autoencoder_history,
        "unet_history": unet_history,
        "summary": summary,
        "test_loader": test_loader,
    }


def print_comparison_summary(summary: dict):
    print("\n=== reconstruction comparison summary (STL-10 test set) ===")
    header = f"{'model':<12}{'params':>12}{'test_mse':>12}{'test_psnr_db':>14}{'test_ssim':>12}"
    print(header)
    for name in ("autoencoder", "unet"):
        row = summary[name]
        print(
            f"{name:<12}{row['n_params']:>12,}{row['test_mse']:>12.4f}"
            f"{row['test_psnr_db']:>14.2f}{row['test_ssim']:>12.4f}"
        )


def plot_loss_comparison(autoencoder_history: dict, unet_history: dict, save_path: str | None = None):
    """Overlays both models' train/test MSE curves on one figure."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(autoencoder_history["train_mse"], label="autoencoder (train)", linestyle="--", color="tab:blue")
    ax.plot(autoencoder_history["test_mse"], label="autoencoder (test)", color="tab:blue")
    ax.plot(unet_history["train_mse"], label="U-Net (train)", linestyle="--", color="tab:orange")
    ax.plot(unet_history["test_mse"], label="U-Net (test)", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("pixel-wise MSE")
    ax.set_title("Reconstruction loss: autoencoder vs. U-Net")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved loss comparison figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def visualize_three_way_comparison(
    autoencoder: STL10ConvAutoencoder,
    unet: SmallUNet,
    images: torch.Tensor,
    num_examples: int = 5,
    save_path: str | None = None,
):
    """Row 1 = original, row 2 = autoencoder reconstruction, row 3 =
    U-Net reconstruction, with per-image MSE/PSNR/SSIM (vs. the
    original) under each reconstruction. images: [N,3,96,96] float in
    [0,1], already on the same device as both models."""
    num_examples = min(num_examples, images.size(0))

    autoencoder.eval()
    unet.eval()
    with torch.no_grad():
        originals = images[:num_examples]            # [num_examples,3,96,96]
        autoencoder_recon = autoencoder(originals)     # [num_examples,3,96,96]
        unet_recon = unet(originals)                   # [num_examples,3,96,96]

        def per_image_metrics(recon):
            mse = ((recon - originals) ** 2).mean(dim=(1, 2, 3))  # [num_examples]
            psnr = psnr_from_mse(mse)                              # [num_examples]
            ssim_vals = ssim(recon, originals)                     # [num_examples]
            return mse.cpu(), psnr.cpu(), ssim_vals.cpu()

        ae_mse, ae_psnr, ae_ssim = per_image_metrics(autoencoder_recon)
        unet_mse, unet_psnr, unet_ssim = per_image_metrics(unet_recon)

    originals_cpu = originals.cpu()
    autoencoder_recon_cpu = autoencoder_recon.cpu()
    unet_recon_cpu = unet_recon.cpu()

    fig, axes = plt.subplots(3, num_examples, figsize=(2.3 * num_examples, 6.8))
    if num_examples == 1:
        axes = axes.reshape(3, 1)

    def to_hwc(img_chw):
        return img_chw.permute(1, 2, 0).numpy()  # [3,96,96] -> [96,96,3]

    for i in range(num_examples):
        axes[0, i].imshow(to_hwc(originals_cpu[i]))
        axes[0, i].set_title("original" if i == 0 else "", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(to_hwc(autoencoder_recon_cpu[i]))
        axes[1, i].set_title(
            f"autoencoder\nMSE={ae_mse[i].item():.4f} PSNR={ae_psnr[i].item():.1f}dB SSIM={ae_ssim[i].item():.3f}",
            fontsize=7,
        )
        axes[1, i].axis("off")

        axes[2, i].imshow(to_hwc(unet_recon_cpu[i]))
        axes[2, i].set_title(
            f"U-Net\nMSE={unet_mse[i].item():.4f} PSNR={unet_psnr[i].item():.1f}dB SSIM={unet_ssim[i].item():.3f}",
            fontsize=7,
        )
        axes[2, i].axis("off")

    fig.suptitle(f"original / autoencoder / U-Net reconstruction, {num_examples} examples")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved three-way comparison figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, 1 tiny epoch for each model. Confirms both
    models train without error on the same synthetic loader, both
    evaluation/metric code paths run, and the three-way visualization
    + loss-comparison plotting code paths run."""
    torch.manual_seed(0)
    from torch.utils.data import DataLoader, TensorDataset

    synthetic_images = torch.rand(8, 3, 96, 96, device=device)  # [8,3,96,96], in [0,1]
    synthetic_labels = torch.zeros(8, dtype=torch.long, device=device)  # unused, placeholder
    synthetic_loader = DataLoader(TensorDataset(synthetic_images, synthetic_labels), batch_size=4)

    loss_fn = nn.MSELoss()

    torch.manual_seed(0)
    autoencoder = STL10ConvAutoencoder(latent_dim=32).to(device)
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    autoencoder_history = {"train_mse": [], "test_mse": []}
    for _ in range(2):
        train_mse = train_autoencoder_one_epoch(autoencoder, synthetic_loader, loss_fn, autoencoder_optimizer, device)
        test_mse = evaluate_autoencoder(autoencoder, synthetic_loader, loss_fn, device)
        autoencoder_history["train_mse"].append(train_mse)
        autoencoder_history["test_mse"].append(test_mse)
    assert torch.isfinite(torch.tensor(autoencoder_history["test_mse"][-1])), "autoencoder test_mse not finite"

    torch.manual_seed(0)
    unet = SmallUNet().to(device)
    unet_optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
    unet_history = {"train_mse": [], "test_mse": []}
    for _ in range(2):
        train_mse = train_unet_one_epoch(unet, synthetic_loader, loss_fn, unet_optimizer, device)
        test_mse = evaluate_unet(unet, synthetic_loader, loss_fn, device)
        unet_history["train_mse"].append(train_mse)
        unet_history["test_mse"].append(test_mse)
    assert torch.isfinite(torch.tensor(unet_history["test_mse"][-1])), "unet test_mse not finite"

    summary = {
        "autoencoder": {
            "n_params": count_trainable_parameters(autoencoder),
            "test_mse": autoencoder_history["test_mse"][-1],
            "test_psnr_db": psnr_from_mse(torch.tensor(autoencoder_history["test_mse"][-1])).item(),
            "test_ssim": evaluate_autoencoder_ssim(autoencoder, synthetic_loader, device),
        },
        "unet": {
            "n_params": count_trainable_parameters(unet),
            "test_mse": unet_history["test_mse"][-1],
            "test_psnr_db": psnr_from_mse(torch.tensor(unet_history["test_mse"][-1])).item(),
            "test_ssim": evaluate_unet_ssim(unet, synthetic_loader, device),
        },
    }
    print_comparison_summary(summary)

    plot_loss_comparison(autoencoder_history, unet_history, save_path="compare_smoke_loss.png")
    visualize_three_way_comparison(autoencoder, unet, synthetic_images, num_examples=4, save_path="compare_smoke_three_way.png")

    print("\nsmoke test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="run lightweight local verification with synthetic tensors, no download")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128, help="autoencoder bottleneck size")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    results = run_comparison(
        args.data_root, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        latent_dim=args.latent_dim, device=device, seed=args.seed,
    )
    print_comparison_summary(results["summary"])
    plot_loss_comparison(
        results["autoencoder_history"], results["unet_history"],
        save_path="stl10_autoencoder_vs_unet_loss.png",
    )

    sample_images, _sample_labels = next(iter(results["test_loader"]))
    sample_images = sample_images.to(device)
    visualize_three_way_comparison(
        results["autoencoder"], results["unet"], sample_images, num_examples=5,
        save_path="stl10_autoencoder_vs_unet_reconstructions.png",
    )


if __name__ == "__main__":
    main()
