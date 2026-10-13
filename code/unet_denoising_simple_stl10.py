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

Cross-noise-level evaluation (the addition over the plain version):
the model is trained at ONE fixed noise_std (the "source" std, the
same --noise-std as before -- training itself is unchanged), but is
then TESTED at a whole array of different, fixed noise_std values
spanning [0,1], and every sweep point's result is reported relative to
the model's own result AT ITS SOURCE STD:

    mse_ratio_vs_source(sigma)   = test_mse(sigma)  / test_mse(source_std)
    delta_psnr_vs_source(sigma)  = test_psnr(sigma) - test_psnr(source_std)

This directly answers "how much worse (or better) does the network do
on noise levels it wasn't trained for, compared to the noise level it
WAS trained for" -- e.g. a ratio of 1.0 at sigma=source_std by
definition, rising above 1.0 as sigma moves away from it. See
`evaluate_std_sweep` (quantitative, whole test set) and
`visualize_std_sweep_examples` (qualitative, a few example images).

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


@torch.no_grad()
def evaluate_std_sweep(model, loader, loss_fn, device, std_values):
    """Test-set denoising MSE at each FIXED sigma in std_values (the
    whole test set is corrupted at that one sigma per sweep point --
    unrelated to whatever single noise_std the model was trained at).
    Returns {sigma: average_test_mse}."""
    model.eval()
    sweep_mse = {}

    for std in std_values:
        total_loss, total_samples = 0.0, 0
        for clean_images, _labels in loader:
            clean_images = clean_images.to(device)
            noisy_images = add_gaussian_noise(clean_images, std)
            x_hat = model(noisy_images)
            loss = loss_fn(x_hat, clean_images)
            total_loss += loss.item() * clean_images.size(0)
            total_samples += clean_images.size(0)
        sweep_mse[std] = total_loss / total_samples

    return sweep_mse


def summarize_std_sweep_relative_to_source(sweep_mse: dict, source_std: float) -> dict:
    """Turns {sigma: mse} into {sigma: {mse, psnr_db, mse_ratio_vs_source,
    delta_psnr_db_vs_source}}, where "source" = the noise_std the model
    was actually TRAINED at. source_std must already be a key of
    sweep_mse (i.e. included in the swept std_values) -- it is the
    baseline every other sigma's result is measured against:

        mse_ratio_vs_source(sigma)  = mse(sigma)  / mse(source_std)   (1.0 at sigma=source_std)
        delta_psnr_vs_source(sigma) = psnr(sigma) - psnr(source_std)  (0.0 dB at sigma=source_std)
    """
    assert source_std in sweep_mse, (
        f"source_std={source_std} must be one of the swept std_values {sorted(sweep_mse)} "
        f"so its own result can be used as the baseline"
    )
    source_mse = sweep_mse[source_std]
    source_psnr_db = psnr_from_mse(torch.tensor(source_mse)).item()

    summary = {}
    for std, mse in sweep_mse.items():
        psnr_db = psnr_from_mse(torch.tensor(mse)).item()
        summary[std] = {
            "mse": mse,
            "psnr_db": psnr_db,
            "mse_ratio_vs_source": mse / source_mse,
            "delta_psnr_db_vs_source": psnr_db - source_psnr_db,
        }
    return summary


def print_std_sweep_summary(summary: dict, source_std: float):
    print(f"\n=== denoising performance vs. noise_std, relative to source_std={source_std:.2f} ===")
    header = f"{'sigma':>8}{'mse':>12}{'psnr_db':>12}{'mse_ratio_vs_source':>22}{'delta_psnr_vs_source':>22}"
    print(header)
    for std in sorted(summary):
        row = summary[std]
        marker = "  <- source" if std == source_std else ""
        print(
            f"{std:>8.2f}{row['mse']:>12.4f}{row['psnr_db']:>12.2f}"
            f"{row['mse_ratio_vs_source']:>22.3f}{row['delta_psnr_db_vs_source']:>22.2f}{marker}"
        )


def plot_std_sweep(summary: dict, source_std: float, save_path: str | None = None):
    """Two-panel figure: left = test_denoising_mse vs. test-time sigma
    (absolute units), right = mse_ratio_vs_source vs. sigma (the
    relative, apples-to-apples comparison against the model's own
    result at the std it was trained on). The source_std point is
    marked in both panels."""
    std_values = sorted(summary.keys())
    mse_values = [summary[std]["mse"] for std in std_values]
    ratio_values = [summary[std]["mse_ratio_vs_source"] for std in std_values]

    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_abs.plot(std_values, mse_values, marker="o", color="tab:blue")
    ax_abs.axvline(source_std, color="tab:red", linestyle="--", label=f"source_std={source_std:.2f}")
    ax_abs.set_xlabel("test-time noise_std (sigma)")
    ax_abs.set_ylabel("test_denoising_mse")
    ax_abs.set_title("Absolute denoising MSE")
    ax_abs.legend()

    ax_rel.plot(std_values, ratio_values, marker="o", color="tab:orange")
    ax_rel.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax_rel.axvline(source_std, color="tab:red", linestyle="--", label=f"source_std={source_std:.2f}")
    ax_rel.set_xlabel("test-time noise_std (sigma)")
    ax_rel.set_ylabel("MSE(sigma) / MSE(source_std)")
    ax_rel.set_title("MSE relative to the source (training) std")
    ax_rel.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved std-sweep figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


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


def visualize_std_sweep_examples(
    model: SmallUNet,
    clean_images: torch.Tensor,
    std_values,
    source_std: float,
    num_examples: int = 3,
    save_path: str | None = None,
):
    """Qualitative check for a few images: row 0 = clean, one row per
    sigma in std_values = that sigma's SmallUNet output, with per-image
    MSE/PSNR vs. clean AND, for every row except the source_std row
    itself, delta_mse = MSE(sigma) - MSE(source_std) FOR THAT SAME
    IMAGE -- so each column directly shows how much worse (positive
    delta) or better (negative delta) denoising gets at sigma compared
    to the std the network was actually trained on. The source_std row
    is labeled "[source]" and used as that column's own baseline (not
    an aggregate/test-set-average baseline). std_values should include
    source_std. clean_images: [N,3,96,96] float in [0,1], already on
    model's device."""
    num_examples = min(num_examples, clean_images.size(0))
    clean = clean_images[:num_examples]  # [num_examples,3,96,96]

    def to_hwc(img_chw):
        return img_chw.permute(1, 2, 0).cpu().numpy()  # [3,96,96] -> [96,96,3]

    n_rows = 1 + len(std_values)
    fig, axes = plt.subplots(n_rows, num_examples, figsize=(2.2 * num_examples, 2.3 * n_rows))
    if num_examples == 1:
        axes = axes.reshape(n_rows, 1)

    for i in range(num_examples):
        axes[0, i].imshow(to_hwc(clean[i]))
        axes[0, i].set_title("clean (target)" if i == 0 else "", fontsize=9)
        axes[0, i].axis("off")

    model.eval()
    with torch.no_grad():
        # source_std's own per-image MSE, used as every column's baseline
        source_noisy = add_gaussian_noise(clean, source_std)
        source_mse = ((model(source_noisy) - clean) ** 2).mean(dim=(1, 2, 3))  # [num_examples]

        for row, std in enumerate(sorted(std_values), start=1):
            noisy = add_gaussian_noise(clean, std)                        # [num_examples,3,96,96]
            x_hat = model(noisy)                                          # [num_examples,3,96,96]
            per_image_mse = ((x_hat - clean) ** 2).mean(dim=(1, 2, 3))     # [num_examples]
            per_image_psnr = psnr_from_mse(per_image_mse)                  # [num_examples]
            per_image_delta_mse = per_image_mse - source_mse               # [num_examples], vs. this image's own source_std result
            is_source = (std == source_std)

            for i in range(num_examples):
                axes[row, i].imshow(to_hwc(x_hat[i]))
                if is_source:
                    title = f"sigma={std:.2f} [source]\nMSE={per_image_mse[i].item():.4f} PSNR={per_image_psnr[i].item():.1f}dB"
                else:
                    title = (
                        f"sigma={std:.2f}\nMSE={per_image_mse[i].item():.4f} PSNR={per_image_psnr[i].item():.1f}dB\n"
                        f"vs. source: {per_image_delta_mse[i].item():+.4f}"
                    )
                axes[row, i].set_title(title, fontsize=7)
                axes[row, i].axis("off")

    fig.suptitle(f"Denoising vs. test-time noise_std, relative to source_std={source_std:.2f}, {num_examples} examples")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved std-sweep qualitative figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, 2 tiny epochs. Confirms construction, one
    training epoch, one evaluation pass, finite loss, the plain
    denoising-visualization code path, and the std-sweep-relative-to-
    source code path (quantitative table + both figures) at STL-10's
    native 96x96 resolution."""
    torch.manual_seed(0)
    from torch.utils.data import DataLoader, TensorDataset

    model = SmallUNet().to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    synthetic_images = torch.rand(8, 3, 96, 96, device=device)  # [8,3,96,96], in [0,1]
    synthetic_labels = torch.zeros(8, dtype=torch.long, device=device)  # unused, placeholder
    synthetic_loader = DataLoader(TensorDataset(synthetic_images, synthetic_labels), batch_size=4)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    source_std = 0.3  # the std the model is trained at

    for epoch in range(2):
        train_loss = train_one_epoch(model, synthetic_loader, loss_fn, optimizer, device, source_std)
        test_mse = evaluate(model, synthetic_loader, loss_fn, device, source_std)
        print(f"epoch {epoch + 1}/2  train_loss={train_loss:.4f}  test_denoising_mse={test_mse:.4f}")
    assert torch.isfinite(torch.tensor(test_mse)), "test_denoising_mse is not finite"

    scratch_png = "unet_denoising_simple_stl10_smoke.png"
    visualize_denoising(model, synthetic_images, source_std, num_examples=4, save_path=scratch_png)

    std_sweep_values = sorted({0.0, 0.3, 0.5, 1.0} | {source_std})  # source_std must be included
    sweep_mse = evaluate_std_sweep(model, synthetic_loader, loss_fn, device, std_sweep_values)
    summary = summarize_std_sweep_relative_to_source(sweep_mse, source_std)
    print_std_sweep_summary(summary, source_std)
    assert summary[source_std]["mse_ratio_vs_source"] == 1.0, "source_std's own ratio must be exactly 1.0"
    assert all(torch.isfinite(torch.tensor(row["mse"])) for row in summary.values()), "sweep MSE not finite"

    scratch_png_sweep_curve = "unet_denoising_simple_stl10_smoke_sweep_curve.png"
    plot_std_sweep(summary, source_std, save_path=scratch_png_sweep_curve)

    scratch_png_qualitative = "unet_denoising_simple_stl10_smoke_qualitative.png"
    visualize_std_sweep_examples(
        model, synthetic_images, std_sweep_values, source_std, num_examples=3, save_path=scratch_png_qualitative,
    )

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
                         help="std of the additive Gaussian noise the model is TRAINED at "
                              "(the 'source' std the sweep is measured relative to)")
    parser.add_argument("--std-sweep-values", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0",
                         help="comma-separated FIXED noise_std values to test the trained model at "
                              "(--noise-std is added automatically if not already present)")
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

    # source_std (the training std) must be in the sweep -- it is the baseline every other point is measured against
    std_sweep_values = sorted(set(float(v) for v in args.std_sweep_values.split(",")) | {args.noise_std})
    sweep_mse = evaluate_std_sweep(model, test_loader, loss_fn, device, std_sweep_values)
    summary = summarize_std_sweep_relative_to_source(sweep_mse, args.noise_std)
    print_std_sweep_summary(summary, args.noise_std)
    plot_std_sweep(summary, args.noise_std, save_path="unet_denoising_simple_stl10_std_sweep_curve.png")

    visualize_std_sweep_examples(
        model, sample_images, std_sweep_values, args.noise_std, num_examples=3,
        save_path="unet_denoising_simple_stl10_std_sweep_qualitative.png",
    )


if __name__ == "__main__":
    main()
