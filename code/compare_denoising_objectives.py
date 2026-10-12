"""Denoising objective comparison: predict the clean image (x0) vs.
predict the added noise (eps), both using the identical TinyUNet64
architecture (code/unet_tiny_denoising_64.py) on the identical task
(CIFAR-10 resized to 64x64, additive Gaussian noise, same noise_std),
trained with identical hyperparameters (same seed at construction, same
epochs/batch size/learning rate/optimizer, same train/test split).

Two objectives compared:
    "x0"    L = ||x - UNet(x_noisy)||^2           (network outputs x_hat directly)
    "noise" L = ||eps - UNet(x_noisy)||^2          (network outputs eps_hat;
                                                     x_hat = clamp(x_noisy - eps_hat, 0, 1) afterward)
where x_noisy = x + sigma*z (z ~ N(0,I), sigma = noise_std) and
eps := x_noisy - x is the actual (already-clamped) corruption applied
-- see unet_tiny_denoising_64.py::make_noisy_and_target_noise for why
this definition keeps the x0/eps algebra exact.

Because the two objectives' raw training losses live in different
units (pixel-space MSE vs. noise-space MSE), they are NOT directly
comparable as numbers. The fair, common metric used throughout this
script is `evaluate_denoising_mse`: the MSE of the *recovered clean-
image estimate x_hat* against the true clean image, which is in
identical units ([0,1] pixel space) for both objectives.

What is controlled: random seed (reset immediately before each model's
construction, so both start from bit-identical initial weights -- the
architecture's parameters do not depend on prediction_mode, only the
final activation applied in forward() does), epochs, batch size,
learning rate, optimizer (Adam), noise_std, and the dataset/train-test
split (the same two DataLoaders are reused for both models). What is
NOT controlled: the exact Gaussian-noise draws and mini-batch order
during each model's own training loop, since both consume the shared
DataLoader's shuffling RNG and torch's global RNG independently once
their training starts (same caveat as
compare_stl10_autoencoder_vs_unet.py; see CLAUDE.md Research
Integrity).

Local usage (lightweight smoke test only, no dataset download):
    python code/compare_denoising_objectives.py --smoke-test

Colab usage (real data, full training of both models):
    python code/compare_denoising_objectives.py --data-root /content/data \
        --epochs 10 --batch-size 128 --lr 1e-3 --noise-std 0.3
"""

import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from unet_tiny_denoising_64 import (
    TinyUNet64,
    count_trainable_parameters,
    psnr_from_mse,
    add_gaussian_noise,
    denoise_to_clean_image,
    get_cifar10_denoising_dataloaders,
    train_one_epoch,
    evaluate,
    evaluate_denoising_mse,
)

PREDICTION_MODES = ("x0", "noise")


def run_comparison(
    data_root: str,
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    noise_std: float = 0.3,
    base_channels: int = 32,
    device: torch.device | None = None,
    seed: int = 0,
):
    """Trains one TinyUNet64 per prediction_mode in PREDICTION_MODES,
    from scratch, on the same train/test loaders. Returns a dict with
    both trained models, both loss histories (native loss + the common
    denoising-MSE metric, per epoch), both final test metrics, and the
    test_loader (for later visualization)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_denoising_dataloaders(data_root, batch_size, image_size=64)
    loss_fn = nn.MSELoss()

    models, histories = {}, {}
    for prediction_mode in PREDICTION_MODES:
        torch.manual_seed(seed)  # same seed before construction -> identical initial weights
        model = TinyUNet64(
            in_channels=3, out_channels=3, base_channels=base_channels, prediction_mode=prediction_mode,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = {"train_loss_native": [], "test_loss_native": [], "test_denoising_mse": []}

        for epoch in range(epochs):
            train_loss_native = train_one_epoch(model, train_loader, loss_fn, optimizer, device, noise_std)
            test_loss_native = evaluate(model, test_loader, loss_fn, device, noise_std)
            test_denoising_mse = evaluate_denoising_mse(model, test_loader, device, noise_std)

            history["train_loss_native"].append(train_loss_native)
            history["test_loss_native"].append(test_loss_native)
            history["test_denoising_mse"].append(test_denoising_mse)

            print(
                f"[{prediction_mode:<5}] epoch {epoch + 1:3d}/{epochs}  "
                f"train_loss(native)={train_loss_native:.4f}  "
                f"test_loss(native)={test_loss_native:.4f}  "
                f"test_denoising_mse={test_denoising_mse:.4f}"
            )

        models[prediction_mode] = model
        histories[prediction_mode] = history

    summary = {}
    for prediction_mode in PREDICTION_MODES:
        final_denoising_mse = histories[prediction_mode]["test_denoising_mse"][-1]
        summary[prediction_mode] = {
            "n_params": count_trainable_parameters(models[prediction_mode]),
            "test_loss_native": histories[prediction_mode]["test_loss_native"][-1],
            "test_denoising_mse": final_denoising_mse,
            "test_denoising_psnr_db": psnr_from_mse(torch.tensor(final_denoising_mse)).item(),
        }

    return {
        "models": models,
        "histories": histories,
        "summary": summary,
        "test_loader": test_loader,
        "noise_std": noise_std,
    }


def print_comparison_summary(summary: dict):
    print("\n=== denoising objective comparison summary (CIFAR-10 64x64 test set) ===")
    print("(test_loss_native is in each objective's OWN units -- pixel MSE for x0, noise MSE")
    print(" for noise -- and is NOT comparable across rows; test_denoising_mse/psnr IS comparable,")
    print(" since both are computed on the recovered clean-image estimate x_hat.)")
    header = f"{'mode':<8}{'params':>12}{'test_loss_native':>18}{'test_denoising_mse':>20}{'test_denoising_psnr_db':>24}"
    print(header)
    for prediction_mode in PREDICTION_MODES:
        row = summary[prediction_mode]
        print(
            f"{prediction_mode:<8}{row['n_params']:>12,}{row['test_loss_native']:>18.4f}"
            f"{row['test_denoising_mse']:>20.4f}{row['test_denoising_psnr_db']:>24.2f}"
        )


def plot_loss_comparison(histories: dict, save_path: str | None = None):
    """Two-panel figure: left = each objective's own native train/test
    loss (own units -- shows each model is converging, but the two
    curves are NOT on a shared scale); right = the common,
    directly-comparable metric (test_denoising_mse per epoch), both
    objectives on the same axes -- this is the actual fair comparison.
    """
    fig, (ax_native, ax_common) = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = {"x0": "tab:blue", "noise": "tab:orange"}
    for prediction_mode in PREDICTION_MODES:
        history = histories[prediction_mode]
        color = colors[prediction_mode]
        ax_native.plot(
            history["train_loss_native"], label=f"{prediction_mode} (train)",
            linestyle="--", color=color,
        )
        ax_native.plot(
            history["test_loss_native"], label=f"{prediction_mode} (test)",
            color=color,
        )
    ax_native.set_xlabel("epoch")
    ax_native.set_ylabel("loss (each objective's own units)")
    ax_native.set_title("Native training loss\n(NOT comparable across objectives)")
    ax_native.legend(fontsize=8)

    for prediction_mode in PREDICTION_MODES:
        ax_common.plot(
            histories[prediction_mode]["test_denoising_mse"],
            label=prediction_mode, color=colors[prediction_mode],
        )
    ax_common.set_xlabel("epoch")
    ax_common.set_ylabel("MSE(x_hat, clean)")
    ax_common.set_title("Test denoising MSE\n(comparable: same units for both)")
    ax_common.legend(fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved loss comparison figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def visualize_denoising_comparison(
    models: dict,
    clean_images: torch.Tensor,
    noise_std: float,
    num_examples: int = 5,
    save_path: str | None = None,
):
    """Row 1 = clean (ground truth), row 2 = noisy input (shared across
    both models -- same noise draw), row 3 = x0-model's x_hat, row 4 =
    noise-model's x_hat (via denoise_to_clean_image, so both rows are
    directly comparable pixel estimates), with per-image MSE/PSNR under
    each reconstruction row. clean_images: [N,3,64,64] in [0,1], on the
    same device as both models."""
    num_examples = min(num_examples, clean_images.size(0))

    for model in models.values():
        model.eval()
    with torch.no_grad():
        clean = clean_images[:num_examples]                  # [num_examples,3,64,64]
        noisy = add_gaussian_noise(clean, noise_std)          # [num_examples,3,64,64], shared by both models

        def per_image_metrics(x_hat):
            mse = ((x_hat - clean) ** 2).mean(dim=(1, 2, 3))  # [num_examples]
            psnr = psnr_from_mse(mse)                          # [num_examples]
            return mse.cpu(), psnr.cpu()

        x0_hat = denoise_to_clean_image(models["x0"], noisy)
        noise_hat = denoise_to_clean_image(models["noise"], noisy)
        x0_mse, x0_psnr = per_image_metrics(x0_hat)
        noise_mse, noise_psnr = per_image_metrics(noise_hat)

    clean_cpu, noisy_cpu = clean.cpu(), noisy.cpu()
    x0_hat_cpu, noise_hat_cpu = x0_hat.cpu(), noise_hat.cpu()

    def to_hwc(img_chw):
        return img_chw.permute(1, 2, 0).numpy()  # [3,64,64] -> [64,64,3]

    fig, axes = plt.subplots(4, num_examples, figsize=(2.2 * num_examples, 8.6))
    if num_examples == 1:
        axes = axes.reshape(4, 1)

    for i in range(num_examples):
        axes[0, i].imshow(to_hwc(clean_cpu[i]))
        axes[0, i].set_title("clean (target)" if i == 0 else "", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(to_hwc(noisy_cpu[i]))
        axes[1, i].set_title(f"noisy (std={noise_std})" if i == 0 else "", fontsize=9)
        axes[1, i].axis("off")

        axes[2, i].imshow(to_hwc(x0_hat_cpu[i]))
        axes[2, i].set_title(
            f"x0-pred\nMSE={x0_mse[i].item():.4f} PSNR={x0_psnr[i].item():.1f}dB", fontsize=8,
        )
        axes[2, i].axis("off")

        axes[3, i].imshow(to_hwc(noise_hat_cpu[i]))
        axes[3, i].set_title(
            f"noise-pred\nMSE={noise_mse[i].item():.4f} PSNR={noise_psnr[i].item():.1f}dB", fontsize=8,
        )
        axes[3, i].axis("off")

    fig.suptitle(f"clean / noisy / x0-objective x_hat / noise-objective x_hat, {num_examples} examples")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved objective comparison figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, 2 tiny epochs per model. Confirms both objectives
    train without error on the same synthetic loader, the common
    denoising-MSE metric is computed correctly for both, and the
    loss-comparison / qualitative-comparison plotting code paths run.
    """
    torch.manual_seed(0)
    from torch.utils.data import DataLoader, TensorDataset

    synthetic_images = torch.rand(8, 3, 64, 64, device=device)  # [8,3,64,64], in [0,1]
    synthetic_labels = torch.zeros(8, dtype=torch.long, device=device)  # unused, placeholder
    synthetic_loader = DataLoader(TensorDataset(synthetic_images, synthetic_labels), batch_size=4)

    loss_fn = nn.MSELoss()
    noise_std = 0.3

    models, histories = {}, {}
    for prediction_mode in PREDICTION_MODES:
        torch.manual_seed(0)
        model = TinyUNet64(in_channels=3, out_channels=3, base_channels=32, prediction_mode=prediction_mode).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = {"train_loss_native": [], "test_loss_native": [], "test_denoising_mse": []}
        for _ in range(2):
            train_loss_native = train_one_epoch(model, synthetic_loader, loss_fn, optimizer, device, noise_std)
            test_loss_native = evaluate(model, synthetic_loader, loss_fn, device, noise_std)
            test_denoising_mse = evaluate_denoising_mse(model, synthetic_loader, device, noise_std)
            history["train_loss_native"].append(train_loss_native)
            history["test_loss_native"].append(test_loss_native)
            history["test_denoising_mse"].append(test_denoising_mse)
        assert torch.isfinite(torch.tensor(history["test_denoising_mse"][-1])), (
            f"{prediction_mode}: test_denoising_mse not finite"
        )
        models[prediction_mode] = model
        histories[prediction_mode] = history

    summary = {}
    for prediction_mode in PREDICTION_MODES:
        final_denoising_mse = histories[prediction_mode]["test_denoising_mse"][-1]
        summary[prediction_mode] = {
            "n_params": count_trainable_parameters(models[prediction_mode]),
            "test_loss_native": histories[prediction_mode]["test_loss_native"][-1],
            "test_denoising_mse": final_denoising_mse,
            "test_denoising_psnr_db": psnr_from_mse(torch.tensor(final_denoising_mse)).item(),
        }
    print_comparison_summary(summary)

    plot_loss_comparison(histories, save_path="compare_denoising_smoke_loss.png")
    visualize_denoising_comparison(
        models, synthetic_images, noise_std, num_examples=4, save_path="compare_denoising_smoke_qualitative.png",
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

    results = run_comparison(
        args.data_root, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        noise_std=args.noise_std, base_channels=args.base_channels, device=device, seed=args.seed,
    )
    print_comparison_summary(results["summary"])
    plot_loss_comparison(results["histories"], save_path="denoising_objectives_loss.png")

    sample_images, _sample_labels = next(iter(results["test_loader"]))
    sample_images = sample_images.to(device)
    visualize_denoising_comparison(
        results["models"], sample_images, results["noise_std"], num_examples=5,
        save_path="denoising_objectives_reconstructions.png",
    )


if __name__ == "__main__":
    main()
