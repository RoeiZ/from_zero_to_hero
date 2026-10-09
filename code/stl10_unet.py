"""Small U-Net for STL-10 (96x96x3 RGB natural images), built from plain
torch.nn layers only (no external U-Net/model library: no `diffusers`,
no `segmentation_models_pytorch`, etc.).

This is the first skip-connection architecture in the repo, meant to be
compared directly against code/stl10_conv_autoencoder.py on the same
task (pixel-wise reconstruction of the input image):

    L(x) = || x - unet(x) ||^2        (pixel-wise MSE, identical loss)

The key architectural difference from the plain autoencoder is *where
information is allowed to flow*. The autoencoder squeezes the whole
image through a single FC bottleneck vector z in R^128, so every detail
not captured by those 128 numbers is permanently lost. The U-Net instead
keeps one skip connection per resolution level: the *full* encoder
feature map at each spatial size (96x96, 48x48, 24x24) is concatenated
(channel-wise) with the decoder feature map at that same size, so
fine-grained spatial detail (edges, textures) can bypass the bottleneck
entirely instead of being forced through it. The 12x12 bottleneck here
is a feature map, not a flattened vector, so there is no FC layer at
all -- the network is fully convolutional.

Architecture: 3 encoder levels (96 -> 48 -> 24 -> 12 via stride-2 convs)
with channels 32 -> 64 -> 128 -> 256, a convolutional bottleneck at
12x12, and 3 decoder levels mirroring back up to 96x96, each decoder
level concatenating its upsampled feature map with the matching
encoder skip before a fusing conv. Every encoder/decoder op is a named
layer (no nn.Sequential), and forward() prints the [B,C,H,W] shape
after every single op when verbose=True -- including, per level, the
shape of the skip tensor at the moment it is saved, the shapes of both
tensors immediately *before* each `torch.cat`, and the fused shape
immediately *after* it.

Shape trace for input [B, 3, 96, 96]:

    --- encoder ---
    enc0_conv (3  ->32 , k3,s1,p1) -> [B, 32, 96, 96]         (-> skip0)
    down0     (32 ->64 , k3,s2,p1) -> [B, 64, 48, 48]
    enc1_conv (64 ->64 , k3,s1,p1) -> [B, 64, 48, 48]         (-> skip1)
    down1     (64 ->128, k3,s2,p1) -> [B,128, 24, 24]
    enc2_conv (128->128, k3,s1,p1) -> [B,128, 24, 24]         (-> skip2)
    down2     (128->256, k3,s2,p1) -> [B,256, 12, 12]

    --- bottleneck ---
    bottleneck_conv (256->256, k3,s1,p1) -> [B,256, 12, 12]   (no FC layer)

    --- decoder ---
    up2       (256->128, k3,s2,p1,op1) -> [B,128, 24, 24]
    cat([up2, skip2], dim=1)           -> [B,256, 24, 24]     (128+128 channels)
    dec2_conv (256->128, k3,s1,p1)     -> [B,128, 24, 24]
    up1       (128->64 , k3,s2,p1,op1) -> [B, 64, 48, 48]
    cat([up1, skip1], dim=1)           -> [B,128, 48, 48]     (64+64 channels)
    dec1_conv (128->64 , k3,s1,p1)     -> [B, 64, 48, 48]
    up0       (64 ->32 , k3,s2,p1,op1) -> [B, 32, 96, 96]
    cat([up0, skip0], dim=1)           -> [B, 64, 96, 96]     (32+32 channels)
    dec0_conv (64 ->32 , k3,s1,p1)     -> [B, 32, 96, 96]
    out_conv  (32 ->3  , k1,s1,p0)     -> [B,  3, 96, 96]
    sigmoid                            -> [B,  3, 96, 96]     (x_hat in [0,1])

Stride-2 3x3 convolutions with padding=1 halve H,W exactly, following
out = floor((in + 2*pad - kernel) / stride) + 1 (e.g. (96+2-3)/2+1 = 48);
96 is divisible by 8 (2^3 downsampling stages here), so every halving
lands on an even number (96->48->24->12) with no rounding ambiguity.
Each decoder ConvTranspose2d uses output_padding=1 to recover the exact
2x upsampling, via out = (in-1)*stride - 2*pad + kernel + output_padding,
e.g. up2: (12-1)*2-2+3+1 = 22-2+3+1 = 24, matching skip2's spatial size
exactly -- this is required for `torch.cat` to succeed (concatenation
along the channel dim requires identical H,W on both tensors).

Local usage (lightweight smoke test only, no dataset download):
    python code/stl10_unet.py --smoke-test

Colab usage (real data, full training). STL-10 is a ~2.6GB download, so
this must run on Colab with Drive caching, not locally:
    python code/stl10_unet.py --data-root /content/data \
        --epochs 15 --batch-size 64 --lr 1e-3
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SmallUNet(nn.Module):
    """3-level U-Net (96->48->24->12, channels 32->64->128->256) with a
    convolutional bottleneck and 3 skip connections. See module
    docstring for the full shape trace and the math/shape reasoning.
    """

    def __init__(self):
        super().__init__()

        # --- encoder level 0: 96x96, channels 3->32 ---
        self.enc0_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.enc0_relu = nn.ReLU()
        self.down0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.down0_relu = nn.ReLU()

        # --- encoder level 1: 48x48, channels 64->64 ---
        self.enc1_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.enc1_relu = nn.ReLU()
        self.down1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.down1_relu = nn.ReLU()

        # --- encoder level 2: 24x24, channels 128->128 ---
        self.enc2_conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.enc2_relu = nn.ReLU()
        self.down2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.down2_relu = nn.ReLU()

        # --- bottleneck: 12x12, channels 256->256 (fully convolutional, no FC) ---
        self.bottleneck_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bottleneck_relu = nn.ReLU()

        # --- decoder level 2: 12x12 -> 24x24, 256->128, fuse with skip2 (128 ch) ---
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2_relu = nn.ReLU()
        self.dec2_conv = nn.Conv2d(in_channels=128 + 128, out_channels=128, kernel_size=3, padding=1)
        self.dec2_relu = nn.ReLU()

        # --- decoder level 1: 24x24 -> 48x48, 128->64, fuse with skip1 (64 ch) ---
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1_relu = nn.ReLU()
        self.dec1_conv = nn.Conv2d(in_channels=64 + 64, out_channels=64, kernel_size=3, padding=1)
        self.dec1_relu = nn.ReLU()

        # --- decoder level 0: 48x48 -> 96x96, 64->32, fuse with skip0 (32 ch) ---
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up0_relu = nn.ReLU()
        self.dec0_conv = nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, padding=1)
        self.dec0_relu = nn.ReLU()

        # --- output head: 32 channels -> 3 RGB channels, same spatial size ---
        self.out_conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """x: [B, 3, 96, 96] in [0,1] -> x_hat: [B, 3, 96, 96] in [0,1]."""

        def step(layer, h, name):
            h = layer(h)
            if verbose:
                print(f"after {name:<16}: {tuple(h.shape)}")
            return h

        def save_skip(h, name):
            if verbose:
                print(f"saved {name:<16}: {tuple(h.shape)}  (kept for later skip connection)")
            return h

        def concat_skip(h, skip, name):
            if verbose:
                print(f"before {name:<15}: upsampled={tuple(h.shape)}  skip={tuple(skip.shape)}")
            h = torch.cat([h, skip], dim=1)  # [B,C_up,H,W] + [B,C_skip,H,W] -> [B,C_up+C_skip,H,W]
            if verbose:
                print(f"after  {name:<15}: {tuple(h.shape)}")
            return h

        if verbose:
            print(f"input                : {tuple(x.shape)}")

        # --- encoder level 0 (96x96) ---
        h = step(self.enc0_conv, x, "enc0_conv")
        h = step(self.enc0_relu, h, "enc0_relu")
        skip0 = save_skip(h, "skip0")
        h = step(self.down0, h, "down0")
        h = step(self.down0_relu, h, "down0_relu")

        # --- encoder level 1 (48x48) ---
        h = step(self.enc1_conv, h, "enc1_conv")
        h = step(self.enc1_relu, h, "enc1_relu")
        skip1 = save_skip(h, "skip1")
        h = step(self.down1, h, "down1")
        h = step(self.down1_relu, h, "down1_relu")

        # --- encoder level 2 (24x24) ---
        h = step(self.enc2_conv, h, "enc2_conv")
        h = step(self.enc2_relu, h, "enc2_relu")
        skip2 = save_skip(h, "skip2")
        h = step(self.down2, h, "down2")
        h = step(self.down2_relu, h, "down2_relu")

        # --- bottleneck (12x12) ---
        h = step(self.bottleneck_conv, h, "bottleneck_conv")
        h = step(self.bottleneck_relu, h, "bottleneck_relu")

        # --- decoder level 2 (-> 24x24), fuse with skip2 ---
        h = step(self.up2, h, "up2")
        h = step(self.up2_relu, h, "up2_relu")
        h = concat_skip(h, skip2, "cat2")
        h = step(self.dec2_conv, h, "dec2_conv")
        h = step(self.dec2_relu, h, "dec2_relu")

        # --- decoder level 1 (-> 48x48), fuse with skip1 ---
        h = step(self.up1, h, "up1")
        h = step(self.up1_relu, h, "up1_relu")
        h = concat_skip(h, skip1, "cat1")
        h = step(self.dec1_conv, h, "dec1_conv")
        h = step(self.dec1_relu, h, "dec1_relu")

        # --- decoder level 0 (-> 96x96), fuse with skip0 ---
        h = step(self.up0, h, "up0")
        h = step(self.up0_relu, h, "up0_relu")
        h = concat_skip(h, skip0, "cat0")
        h = step(self.dec0_conv, h, "dec0_conv")
        h = step(self.dec0_relu, h, "dec0_relu")

        h = step(self.out_conv, h, "out_conv")
        x_hat = step(self.sigmoid, h, "sigmoid")

        return x_hat


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio, PSNR = 10*log10(MAX^2 / MSE), in dB.
    See code/stl10_conv_autoencoder.py::psnr_from_mse for the full
    rationale; identical formula, duplicated here so this script stays
    self-contained (no cross-file imports between exercises)."""
    return 10.0 * torch.log10((max_val ** 2) / (mse + eps))


def _gaussian_ssim_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    """Depthwise Gaussian window for `ssim`. See
    code/stl10_conv_autoencoder.py::_gaussian_ssim_window for details."""
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2  # [window_size]
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_1d = g_1d / g_1d.sum()  # [window_size], sums to 1

    g_2d = g_1d.unsqueeze(1) @ g_1d.unsqueeze(0)  # [window_size,1] @ [1,window_size] -> [window_size,window_size]
    window = g_2d.expand(channels, 1, window_size, window_size).contiguous()  # [C,1,win,win]
    return window


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> torch.Tensor:
    """Structural Similarity Index (Wang et al., 2004), single-scale,
    11x11 Gaussian window (sigma=1.5). Identical implementation to
    code/stl10_conv_autoencoder.py::ssim, duplicated for this script's
    self-containedness. x, y: [B,C,H,W] in [0,max_val] -> SSIM per
    image, [B]."""
    assert x.shape == y.shape, f"x.shape {tuple(x.shape)} != y.shape {tuple(y.shape)}"
    channels = x.shape[1]
    window = _gaussian_ssim_window(window_size, sigma=1.5, channels=channels, device=x.device, dtype=x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=channels)  # [B,C,H,W]
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )  # [B,C,H,W]

    return ssim_map.mean(dim=(1, 2, 3))  # [B]


def get_stl10_dataloaders(data_root: str, batch_size: int):
    """Real STL-10 train/test split (torchvision's native labeled
    'train'/'test' partition; the separate 100k-image 'unlabeled' split
    is not used, same rationale as code/stl10_conv_autoencoder.py).
    Downloads ~2.6GB on first call. Intended for Colab, not local
    smoke tests."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.ToTensor()  # [0,255] uint8 HWC -> [0,1] float32 [3,96,96]

    stl10_train_dataset = torchvision.datasets.STL10(
        root=data_root, split="train", download=True, transform=transform,
    )
    stl10_test_dataset = torchvision.datasets.STL10(
        root=data_root, split="test", download=True, transform=transform,
    )

    train_loader = DataLoader(stl10_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(stl10_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def visualize_reconstructions(
    model: SmallUNet,
    images: torch.Tensor,
    num_examples: int = 5,
    save_path: str | None = None,
):
    """Qualitative + quantitative check: top row = original images,
    bottom row = the U-Net's reconstruction, with per-image MSE, PSNR,
    SSIM in each bottom title. images: [N,3,96,96] float in [0,1],
    already on the same device as `model`."""
    num_examples = min(num_examples, images.size(0))

    model.eval()
    with torch.no_grad():
        originals = images[:num_examples]            # [num_examples, 3, 96, 96]
        reconstructions = model(originals)            # [num_examples, 3, 96, 96]
        per_image_mse = ((reconstructions - originals) ** 2).mean(dim=(1, 2, 3))  # [num_examples]
        per_image_psnr = psnr_from_mse(per_image_mse)  # [num_examples], dB
        per_image_ssim = ssim(reconstructions, originals)  # [num_examples]

    originals_cpu = originals.cpu()
    reconstructions_cpu = reconstructions.cpu()
    per_image_mse_cpu = per_image_mse.cpu()
    per_image_psnr_cpu = per_image_psnr.cpu()
    per_image_ssim_cpu = per_image_ssim.cpu()

    fig, axes = plt.subplots(2, num_examples, figsize=(2.2 * num_examples, 4.6))
    if num_examples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_examples):
        original_hwc = originals_cpu[i].permute(1, 2, 0).numpy()  # [3,96,96] -> [96,96,3]
        reconstruction_hwc = reconstructions_cpu[i].permute(1, 2, 0).numpy()

        axes[0, i].imshow(original_hwc)
        axes[0, i].set_title("original", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstruction_hwc)
        axes[1, i].set_title(
            f"MSE={per_image_mse_cpu[i].item():.4f}\n"
            f"PSNR={per_image_psnr_cpu[i].item():.1f}dB  SSIM={per_image_ssim_cpu[i].item():.3f}",
            fontsize=8,
        )
        axes[1, i].axis("off")

    fig.suptitle(f"original (top) vs. U-Net reconstruction (bottom), {num_examples} examples")
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
        images = images.to(device)  # [B,3,96,96]; label unused, target is the image itself

        optimizer.zero_grad()
        reconstructions = model(images)          # [B,3,96,96]
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


@torch.no_grad()
def evaluate_ssim(model, loader, device, window_size: int = 11):
    """Average SSIM over an entire loader (mirrors `evaluate`'s MSE
    aggregation). Returns a Python float."""
    model.eval()
    total_ssim, total_samples = 0.0, 0

    for images, _labels in loader:
        images = images.to(device)
        reconstructions = model(images)
        batch_ssim = ssim(reconstructions, images, window_size=window_size)  # [B]

        total_ssim += batch_ssim.sum().item()
        total_samples += images.size(0)

    return total_ssim / total_samples


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, no real training. Confirms construction, the
    full forward shape trace (including every skip-save and
    concatenation), parameter count, one backward pass, one optimizer
    step, and the reconstruction-plotting code path."""
    torch.manual_seed(0)

    model = SmallUNet().to(device)
    n_params = count_trainable_parameters(model)
    print(f"trainable parameters: {n_params:,}")

    dummy_images = torch.rand(4, 3, 96, 96, device=device)  # [B,3,96,96], in [0,1] like real STL-10

    print("\n--- forward shape trace (including skip connections and concatenations) ---")
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

    print("\n--- SSIM numerical sanity checks ---")
    probe_images = torch.rand(2, 3, 32, 32, device=device)  # smaller spatial size, just to exercise ssim() cheaply
    self_ssim = ssim(probe_images, probe_images)  # [2]
    assert torch.allclose(self_ssim, torch.ones_like(self_ssim), atol=1e-4), (
        f"SSIM(x,x) should be ~1.0, got {self_ssim.tolist()}"
    )
    print(f"SSIM(x, x)            = {self_ssim.mean().item():.6f} (expected ~1.0, identical images)")

    small_noise = probe_images + 0.02 * torch.randn_like(probe_images)
    large_noise = probe_images + 0.30 * torch.randn_like(probe_images)
    ssim_small_noise = ssim(probe_images, small_noise.clamp(0, 1)).mean().item()
    ssim_large_noise = ssim(probe_images, large_noise.clamp(0, 1)).mean().item()
    assert ssim_small_noise > ssim_large_noise, (
        f"SSIM should drop as noise grows: small_noise={ssim_small_noise:.4f}, large_noise={ssim_large_noise:.4f}"
    )
    print(f"SSIM(x, x+small noise) = {ssim_small_noise:.4f}")
    print(f"SSIM(x, x+large noise) = {ssim_large_noise:.4f}  (lower, as expected)")

    scratch_png = "stl10_unet_smoke_reconstructions.png"
    visualize_reconstructions(model, dummy_images, num_examples=4, save_path=scratch_png)

    print("\nsmoke test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="run lightweight local verification with synthetic tensors, no download")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
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
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss = evaluate(model, test_loader, loss_fn, device)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_mse={train_loss:.4f}  test_mse={test_loss:.4f}")

    test_ssim = evaluate_ssim(model, test_loader, device)
    print(f"final test SSIM: {test_ssim:.4f}")

    sample_images, _sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)
    visualize_reconstructions(
        model, sample_images, num_examples=5,
        save_path="stl10_unet_reconstructions.png",
    )


if __name__ == "__main__":
    main()
