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


def unet_encode(unet: SmallUNet, x: torch.Tensor) -> dict:
    """Runs only SmallUNet's encoder path, by calling its named layers
    directly in the same order as SmallUNet.forward (see
    code/stl10_unet.py) -- no change to that class, just a second entry
    point that stops before the decoder so the three skip tensors and
    the bottleneck feature map can be inspected/swapped independently.

    x: [B,3,96,96] -> dict with skip0 [B,32,96,96], skip1 [B,64,48,48],
    skip2 [B,128,24,24], bottleneck [B,256,12,12]."""
    h = unet.enc0_relu(unet.enc0_conv(x))
    skip0 = h
    h = unet.down0_relu(unet.down0(h))

    h = unet.enc1_relu(unet.enc1_conv(h))
    skip1 = h
    h = unet.down1_relu(unet.down1(h))

    h = unet.enc2_relu(unet.enc2_conv(h))
    skip2 = h
    h = unet.down2_relu(unet.down2(h))

    bottleneck = unet.bottleneck_relu(unet.bottleneck_conv(h))

    return {"skip0": skip0, "skip1": skip1, "skip2": skip2, "bottleneck": bottleneck}


def unet_decode(unet: SmallUNet, encoded: dict, zero_skips: tuple = ()) -> torch.Tensor:
    """Runs only SmallUNet's decoder path given an `encoded` dict from
    `unet_encode` (same bottleneck/skip0/skip1/skip2 keys), optionally
    replacing named skip tensors with zeros before their `torch.cat`
    -- the skip-ablation test: if the reconstruction barely degrades
    with `zero_skips=("skip0","skip1","skip2")`, the bottleneck alone
    already explains most of the output; if it degrades sharply, the
    skip connections were carrying most of the reconstructed detail.

    zero_skips: subset of {"skip0","skip1","skip2"} to zero out."""
    skip0, skip1, skip2 = encoded["skip0"], encoded["skip1"], encoded["skip2"]
    if "skip0" in zero_skips:
        skip0 = torch.zeros_like(skip0)
    if "skip1" in zero_skips:
        skip1 = torch.zeros_like(skip1)
    if "skip2" in zero_skips:
        skip2 = torch.zeros_like(skip2)

    h = unet.up2_relu(unet.up2(encoded["bottleneck"]))
    h = torch.cat([h, skip2], dim=1)  # [B,128,24,24]+[B,128,24,24] -> [B,256,24,24]
    h = unet.dec2_relu(unet.dec2_conv(h))

    h = unet.up1_relu(unet.up1(h))
    h = torch.cat([h, skip1], dim=1)  # [B,64,48,48]+[B,64,48,48] -> [B,128,48,48]
    h = unet.dec1_relu(unet.dec1_conv(h))

    h = unet.up0_relu(unet.up0(h))
    h = torch.cat([h, skip0], dim=1)  # [B,32,96,96]+[B,32,96,96] -> [B,64,96,96]
    h = unet.dec0_relu(unet.dec0_conv(h))

    h = unet.out_conv(h)
    return unet.sigmoid(h)


@torch.no_grad()
def skip_ablation_reconstruction(unet: SmallUNet, images: torch.Tensor) -> torch.Tensor:
    """Skip-ablation test: decode with all three skip tensors zeroed,
    i.e. reconstruct from the [B,256,12,12] bottleneck alone. Compares
    against the normal reconstruction to show how much detail the skip
    connections were responsible for (see module docstring)."""
    unet.eval()
    encoded = unet_encode(unet, images)
    return unet_decode(unet, encoded, zero_skips=("skip0", "skip1", "skip2"))


@torch.no_grad()
def skip_swap_reconstruction(unet: SmallUNet, images_a: torch.Tensor, images_b: torch.Tensor) -> torch.Tensor:
    """Skip-swap test: decode image A's bottleneck using image B's skip
    tensors. If the result looks like B rather than A, the skip
    connections (not the bottleneck) are carrying almost all of the
    reconstructed identity -- the "cheating" failure mode. images_a,
    images_b: [B,3,96,96], same batch size."""
    unet.eval()
    encoded_a = unet_encode(unet, images_a)
    encoded_b = unet_encode(unet, images_b)
    swapped = {
        "bottleneck": encoded_a["bottleneck"],
        "skip0": encoded_b["skip0"],
        "skip1": encoded_b["skip1"],
        "skip2": encoded_b["skip2"],
    }
    return unet_decode(unet, swapped)


def print_shape_traces(device: torch.device, latent_dim: int = 128, batch_size: int = 4):
    """One verbose forward pass per model on a synthetic batch, purely
    to print the full [B,C,H,W] shape trace -- including, for the
    U-Net, every skip-connection save and every concatenation -- before
    training starts. Builds fresh, untrained, throwaway instances of
    both models just for this printout: the real training models built
    inside run_comparison are separate instances with their own seeded
    construction, so this has no effect on the comparison's
    reproducibility."""
    sample_batch = torch.rand(batch_size, 3, 96, 96, device=device)  # [B,3,96,96], synthetic, in [0,1]

    print("=== STL10ConvAutoencoder: full shape trace ===")
    autoencoder_probe = STL10ConvAutoencoder(latent_dim=latent_dim).to(device)
    _ = autoencoder_probe(sample_batch, verbose=True)

    print("\n=== SmallUNet: full shape trace (skip connections + concatenations) ===")
    unet_probe = SmallUNet().to(device)
    _ = unet_probe(sample_batch, verbose=True)


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
    include_skip_diagnostics: bool = False,
):
    """Row 1 = original, row 2 = autoencoder reconstruction, row 3 =
    U-Net reconstruction, with per-image MSE/PSNR/SSIM (vs. the
    original) under each reconstruction. images: [N,3,96,96] float in
    [0,1], already on the same device as both models.

    If include_skip_diagnostics=True, two extra verification rows are
    appended for the U-Net only (the autoencoder has no skip
    connections to test):
        row 4: skip-ablation -- U-Net output with all 3 skip tensors
               zeroed, i.e. reconstructed from the bottleneck alone.
               A reconstruction that stays sharp here would mean the
               bottleneck is doing real work; collapsing to something
               blurry/flat means the skip connections (row 3's normal
               path) were carrying most of the detail.
        row 5: skip-swap -- column i's bottleneck decoded with column
               (i+1)'s skip tensors (cyclic shift). If this row looks
               like the *next* column's original rather than the
               current column's, the skip connections are carrying the
               image's identity almost on their own ("cheating"); see
               skip_ablation_reconstruction / skip_swap_reconstruction.
    """
    num_examples = min(num_examples, images.size(0))

    autoencoder.eval()
    unet.eval()
    with torch.no_grad():
        originals = images[:num_examples]            # [num_examples,3,96,96]
        autoencoder_recon = autoencoder(originals)     # [num_examples,3,96,96]
        unet_recon = unet(originals)                   # [num_examples,3,96,96]

        def per_image_metrics(recon, target):
            mse = ((recon - target) ** 2).mean(dim=(1, 2, 3))  # [num_examples]
            psnr = psnr_from_mse(mse)                           # [num_examples]
            ssim_vals = ssim(recon, target)                     # [num_examples]
            return mse.cpu(), psnr.cpu(), ssim_vals.cpu()

        ae_mse, ae_psnr, ae_ssim = per_image_metrics(autoencoder_recon, originals)
        unet_mse, unet_psnr, unet_ssim = per_image_metrics(unet_recon, originals)

        if include_skip_diagnostics:
            # row 4: bottleneck-only (all skips zeroed), compared to the original
            bottleneck_only_recon = skip_ablation_reconstruction(unet, originals)
            bn_mse, bn_psnr, bn_ssim = per_image_metrics(bottleneck_only_recon, originals)

            # row 5: column i's bottleneck + column (i+1)'s skips (cyclic shift)
            swap_partner_images = originals.roll(shifts=-1, dims=0)  # [num_examples,3,96,96]
            swapped_recon = skip_swap_reconstruction(unet, originals, swap_partner_images)
            # compared to the *swap partner*: a high similarity here is the "cheating" signal
            sw_mse, sw_psnr, sw_ssim = per_image_metrics(swapped_recon, swap_partner_images)

    originals_cpu = originals.cpu()
    autoencoder_recon_cpu = autoencoder_recon.cpu()
    unet_recon_cpu = unet_recon.cpu()

    n_rows = 5 if include_skip_diagnostics else 3
    fig, axes = plt.subplots(n_rows, num_examples, figsize=(2.3 * num_examples, 2.3 * n_rows))
    if num_examples == 1:
        axes = axes.reshape(n_rows, 1)

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

        if include_skip_diagnostics:
            bottleneck_only_recon_cpu = bottleneck_only_recon.cpu()
            swapped_recon_cpu = swapped_recon.cpu()
            swap_partner_index = (i + 1) % num_examples

            axes[3, i].imshow(to_hwc(bottleneck_only_recon_cpu[i]))
            axes[3, i].set_title(
                f"U-Net, skips zeroed\nMSE={bn_mse[i].item():.4f} SSIM={bn_ssim[i].item():.3f}",
                fontsize=7,
            )
            axes[3, i].axis("off")

            axes[4, i].imshow(to_hwc(swapped_recon_cpu[i]))
            axes[4, i].set_title(
                f"U-Net, skips from col {swap_partner_index}\n(vs. col {swap_partner_index}) "
                f"MSE={sw_mse[i].item():.4f} SSIM={sw_ssim[i].item():.3f}",
                fontsize=7,
            )
            axes[4, i].axis("off")

    title = f"original / autoencoder / U-Net reconstruction, {num_examples} examples"
    if include_skip_diagnostics:
        title += "\n+ skip-ablation and skip-swap verification rows (see docstring)"
    fig.suptitle(title)
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

    print("--- shape trace (both models, synthetic batch) ---")
    print_shape_traces(device, latent_dim=32, batch_size=4)

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
    visualize_three_way_comparison(
        autoencoder, unet, synthetic_images, num_examples=4, save_path="compare_smoke_three_way.png",
        include_skip_diagnostics=True,
    )

    print("\n--- skip-ablation / skip-swap sanity checks ---")
    bottleneck_only = skip_ablation_reconstruction(unet, synthetic_images)
    assert bottleneck_only.shape == synthetic_images.shape, (
        f"bottleneck-only reconstruction shape {tuple(bottleneck_only.shape)} != input shape {tuple(synthetic_images.shape)}"
    )
    assert torch.isfinite(bottleneck_only).all(), "bottleneck-only reconstruction has non-finite values"

    swap_partner_images = synthetic_images.roll(shifts=-1, dims=0)
    swapped = skip_swap_reconstruction(unet, synthetic_images, swap_partner_images)
    assert swapped.shape == synthetic_images.shape, (
        f"skip-swap reconstruction shape {tuple(swapped.shape)} != input shape {tuple(synthetic_images.shape)}"
    )
    assert torch.isfinite(swapped).all(), "skip-swap reconstruction has non-finite values"
    print("skip-ablation and skip-swap reconstructions ok (finite, correct shape)")

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

    print_shape_traces(device, latent_dim=args.latent_dim)

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
        include_skip_diagnostics=True,
    )


if __name__ == "__main__":
    main()
