"""Tiny educational U-Net (Ronneberger et al., 2015) for 64x64x3 images,
built from raw torch.nn layers only -- no external model library
(no segmentation_models_pytorch, no monai, no timm).

This is the first encoder/decoder-with-skip-connections architecture in
the thesis path (CNN -> plain autoencoder -> U-Net -> DDPM U-Net), so
the goal here is purely to make the skip-connection bookkeeping
explicit and correct before the same skeleton gets reused as a DDPM
noise-predictor. Every conv/pool/upsample/concat is a named layer (no
nn.Sequential) and forward() prints the [B,C,H,W] shape before AND
after every skip connection is taken and before AND after every
concatenation when verbose=True.

Architecture (3 downsampling stages, input [B,3,64,64]):

    --- encoder ---
    enc1 (3  ->32 , k3,p1) + relu      -> [B, 32, 64, 64]   = skip1
    pool1 (2x2 maxpool)                -> [B, 32, 32, 32]
    enc2 (32 ->64 , k3,p1) + relu      -> [B, 64, 32, 32]   = skip2
    pool2 (2x2 maxpool)                -> [B, 64, 16, 16]
    enc3 (64 ->128, k3,p1) + relu      -> [B,128, 16, 16]   = skip3
    pool3 (2x2 maxpool)                -> [B,128,  8,  8]

    --- bottleneck ---
    bottleneck (128->256, k3,p1) + relu -> [B,256,  8,  8]

    --- decoder ---
    up3 (ConvTranspose2d 256->128,k2,s2) -> [B,128, 16, 16]
    concat(up3, skip3)  [128]+[128]      -> [B,256, 16, 16]
    dec3 (256->128, k3,p1) + relu        -> [B,128, 16, 16]
    up2 (ConvTranspose2d 128->64 ,k2,s2) -> [B, 64, 32, 32]
    concat(up2, skip2)  [64]+[64]        -> [B,128, 32, 32]
    dec2 (128->64 , k3,p1) + relu        -> [B, 64, 32, 32]
    up1 (ConvTranspose2d 64 ->32 ,k2,s2) -> [B, 32, 64, 64]
    concat(up1, skip1)  [32]+[32]        -> [B, 64, 64, 64]
    dec1 (64 ->32 , k3,p1) + relu        -> [B, 32, 64, 64]
    out_conv (32->3, k1)                 -> [B,  3, 64, 64]
    sigmoid                              -> [B,  3, 64, 64]   (in [0,1])

2x2 maxpool exactly halves H,W at every stage since 64 is divisible by
2 three times (64->32->16->8, no rounding). ConvTranspose2d with
kernel=stride=2, padding=0 gives out = (in-1)*stride + kernel = 2*in
exactly, so every upsample recovers the matching encoder resolution and
concatenation along the channel dim (dim=1) never needs cropping --
unlike the original paper's valid-padding convs, whose feature maps
shrink and therefore need center-cropping before concatenation.

Task used to exercise the architecture: image denoising. A clean image
x is corrupted with additive Gaussian noise to make x_noisy, and the
U-Net is trained to map x_noisy -> x_hat ~= x (L = ||x - U-Net(x_noisy)||^2).
This is the same input/output-image-shape setup a DDPM noise-predictor
U-Net uses, so it is a natural next step toward that part of the thesis
-- it is a denoising autoencoder, not a diffusion model (no timestep
conditioning, no noise schedule).

Local usage (lightweight smoke test only, no dataset download):
    python code/unet_tiny_denoising_64.py --smoke-test

Colab usage (real data, full training). Uses CIFAR-10 (small download,
~170MB) upsampled to 64x64 so the U-Net operates at the resolution it
was designed for:
    python code/unet_tiny_denoising_64.py --data-root /content/data \
        --epochs 10 --batch-size 128 --lr 1e-3 --noise-std 0.3
"""

import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class TinyUNet64(nn.Module):
    """3-level encoder/decoder U-Net with skip connections, every op
    named explicitly. See module docstring for the full shape trace.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # --- encoder: 64->32->16->8, channels in->c1->c2->c3 ---
        self.enc1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.relu_e1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.relu_e2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.enc3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.relu_e3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # --- bottleneck: 8x8, channels c3->c4 ---
        self.bottleneck = nn.Conv2d(c3, c4, kernel_size=3, padding=1)
        self.relu_b = nn.ReLU()

        # --- decoder: 8->16->32->64, channels c4->c3->c2->c1 ---
        # kernel=stride=2, padding=0 => out = (in-1)*2 + 2 = 2*in exactly,
        # so the upsampled map always matches its skip tensor's H,W.
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1)  # concat(up3, skip3) -> 2*c3 channels in
        self.relu_d3 = nn.ReLU()
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1)  # concat(up2, skip2) -> 2*c2 channels in
        self.relu_d2 = nn.ReLU()
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1)  # concat(up1, skip1) -> 2*c1 channels in
        self.relu_d1 = nn.ReLU()

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """x: [B, in_channels, 64, 64] -> out: [B, out_channels, 64, 64] in [0,1]."""

        def log(label, tensor):
            if verbose:
                print(f"{label:<42}: {tuple(tensor.shape)}")

        log("input", x)

        # --- encoder ---
        skip1 = self.relu_e1(self.enc1(x))
        log("skip1 saved (before pool1)", skip1)
        h = self.pool1(skip1)
        log("after pool1", h)

        skip2 = self.relu_e2(self.enc2(h))
        log("skip2 saved (before pool2)", skip2)
        h = self.pool2(skip2)
        log("after pool2", h)

        skip3 = self.relu_e3(self.enc3(h))
        log("skip3 saved (before pool3)", skip3)
        h = self.pool3(skip3)
        log("after pool3", h)

        # --- bottleneck ---
        h = self.relu_b(self.bottleneck(h))
        log("after bottleneck", h)

        # --- decoder, stage 3: up3 + concat(skip3) ---
        h = self.up3(h)
        log("up3 output (before concat with skip3)", h)
        log("skip3 (being concatenated)", skip3)
        h = torch.cat([h, skip3], dim=1)  # [B,c3,16,16]+[B,c3,16,16] -> [B,2*c3,16,16] along channel dim
        log("after concat(up3, skip3)", h)
        h = self.relu_d3(self.dec3(h))
        log("after dec3", h)

        # --- decoder, stage 2: up2 + concat(skip2) ---
        h = self.up2(h)
        log("up2 output (before concat with skip2)", h)
        log("skip2 (being concatenated)", skip2)
        h = torch.cat([h, skip2], dim=1)  # [B,c2,32,32]+[B,c2,32,32] -> [B,2*c2,32,32]
        log("after concat(up2, skip2)", h)
        h = self.relu_d2(self.dec2(h))
        log("after dec2", h)

        # --- decoder, stage 1: up1 + concat(skip1) ---
        h = self.up1(h)
        log("up1 output (before concat with skip1)", h)
        log("skip1 (being concatenated)", skip1)
        h = torch.cat([h, skip1], dim=1)  # [B,c1,64,64]+[B,c1,64,64] -> [B,2*c1,64,64]
        log("after concat(up1, skip1)", h)
        h = self.relu_d1(self.dec1(h))
        log("after dec1", h)

        h = self.out_conv(h)
        log("after out_conv", h)
        out = self.sigmoid(h)
        log("after sigmoid (final output)", out)

        return out


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    """PSNR = 10*log10(MAX^2 / MSE), in dB; higher = closer to the clean image.
    max_val=1.0 matches the [0,1] pixel range used throughout (ToTensor
    input, Sigmoid output). eps avoids log(0) for an exact reconstruction."""
    return 10.0 * torch.log10((max_val ** 2) / (mse + eps))


def add_gaussian_noise(images: torch.Tensor, noise_std: float) -> torch.Tensor:
    """images: [B,C,H,W] in [0,1] -> noisy images, same shape, clamped
    back to [0,1] so the U-Net always sees a valid image range."""
    noisy = images + noise_std * torch.randn_like(images)
    return noisy.clamp(0.0, 1.0)


def get_cifar10_denoising_dataloaders(data_root: str, batch_size: int, image_size: int = 64):
    """Real CIFAR-10 train/test split, resized from the native 32x32 up
    to `image_size` (64 by default) so the images match the resolution
    this U-Net was designed for. Labels are unused (the denoising
    target is the clean image itself). Downloads ~170MB on first call;
    intended for Colab, not local smoke tests."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0,255] uint8 HWC -> [0,1] float32 [3,image_size,image_size]
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def visualize_denoising(
    model: TinyUNet64,
    clean_images: torch.Tensor,
    noise_std: float,
    num_examples: int = 5,
    save_path: str | None = None,
):
    """Qualitative check: row 1 = clean (ground truth), row 2 = noisy
    input, row 3 = U-Net output, with per-image MSE/PSNR (output vs.
    clean) in each row-3 title.

    clean_images: [N,C,H,W] float in [0,1], already on model's device.
    """
    num_examples = min(num_examples, clean_images.size(0))

    model.eval()
    with torch.no_grad():
        clean = clean_images[:num_examples]                 # [num_examples,C,H,W]
        noisy = add_gaussian_noise(clean, noise_std)         # [num_examples,C,H,W]
        denoised = model(noisy)                               # [num_examples,C,H,W]
        per_image_mse = ((denoised - clean) ** 2).mean(dim=(1, 2, 3))  # [num_examples]
        per_image_psnr = psnr_from_mse(per_image_mse)          # [num_examples], dB

    clean_cpu, noisy_cpu, denoised_cpu = clean.cpu(), noisy.cpu(), denoised.cpu()
    mse_cpu, psnr_cpu = per_image_mse.cpu(), per_image_psnr.cpu()

    def to_hwc(img_chw):
        return img_chw.permute(1, 2, 0).squeeze(-1).numpy()  # [C,H,W]->[H,W,C], squeeze C=1 for grayscale

    fig, axes = plt.subplots(3, num_examples, figsize=(2.2 * num_examples, 6.6))
    if num_examples == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["clean (target)", "noisy (input)", "U-Net output"]
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

    fig.suptitle(f"clean / noisy (std={noise_std}) / U-Net-denoised, {num_examples} examples")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved denoising figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def train_one_epoch(model, loader, loss_fn, optimizer, device, noise_std):
    model.train()
    total_loss, total_samples = 0.0, 0

    for clean_images, _labels in loader:
        clean_images = clean_images.to(device)              # [B,C,H,W]; label unused
        noisy_images = add_gaussian_noise(clean_images, noise_std)

        optimizer.zero_grad()
        denoised = model(noisy_images)                       # [B,C,H,W]
        loss = loss_fn(denoised, clean_images)                # scalar MSE vs. clean target
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clean_images.size(0)
        total_samples += clean_images.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, noise_std):
    model.eval()
    total_loss, total_samples = 0.0, 0

    for clean_images, _labels in loader:
        clean_images = clean_images.to(device)
        noisy_images = add_gaussian_noise(clean_images, noise_std)
        denoised = model(noisy_images)
        loss = loss_fn(denoised, clean_images)

        total_loss += loss.item() * clean_images.size(0)
        total_samples += clean_images.size(0)

    return total_loss / total_samples


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, no real training. Confirms construction, the
    full forward shape trace (including every skip-connection save and
    every concatenation), parameter count, one backward pass, one
    optimizer step, and the denoising-visualization code path.
    """
    torch.manual_seed(0)

    model = TinyUNet64(in_channels=3, out_channels=3, base_channels=32).to(device)
    n_params = count_trainable_parameters(model)
    print(f"trainable parameters: {n_params:,}")

    dummy_clean = torch.rand(4, 3, 64, 64, device=device)  # [B,3,64,64], in [0,1]
    dummy_noisy = add_gaussian_noise(dummy_clean, noise_std=0.3)

    print("\n--- forward shape trace (every skip connection + concatenation) ---")
    denoised = model(dummy_noisy, verbose=True)
    assert denoised.shape == dummy_clean.shape, (
        f"output shape {tuple(denoised.shape)} != input shape {tuple(dummy_clean.shape)}"
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = loss_fn(denoised, dummy_clean)
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    optimizer.step()
    print(f"\ndummy denoising MSE: {loss.item():.4f} (finite, backward + optimizer step ok)")

    scratch_png = "unet_tiny_denoising_smoke.png"
    visualize_denoising(model, dummy_clean, noise_std=0.3, num_examples=4, save_path=scratch_png)

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

    model = TinyUNet64(in_channels=3, out_channels=3, base_channels=args.base_channels).to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    train_loader, test_loader = get_cifar10_denoising_dataloaders(args.data_root, args.batch_size, image_size=64)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.noise_std)
        test_loss = evaluate(model, test_loader, loss_fn, device, args.noise_std)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_mse={train_loss:.4f}  test_mse={test_loss:.4f}")

    sample_images, _sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)
    visualize_denoising(
        model, sample_images, args.noise_std, num_examples=5,
        save_path="unet_tiny_denoising_reconstructions.png",
    )


if __name__ == "__main__":
    main()
