"""Compact convolutional autoencoder for STL-10 (96x96x3 RGB natural images).

Same architecture pattern as code/mnist_conv_autoencoder.py (explicit
strided-conv encoder, FC bottleneck, transposed-conv decoder), scaled
up with one extra conv/deconv stage and more channels to handle the
larger, visually richer STL-10 images -- the point of this script is
to see reconstruction *quality* (blur, texture loss, color fidelity)
more clearly than MNIST digits allow.

Every encoder/decoder operation is a named layer (no nn.Sequential) and
forward() prints the [B, C, H, W] (or [B, D] in the bottleneck) tensor
shape after each individual operation when verbose=True.

The network minimizes reconstruction error:

    L(x) = || x - decoder(encoder(x)) ||^2        (pixel-wise MSE)

x is the input image itself (no label is used): the encoder compresses
x into a low-dimensional latent vector z = encoder(x) in R^latent_dim,
and the decoder reconstructs x_hat = decoder(z) from z alone. This is
exact mean-squared-error reconstruction loss, not an approximation.

Reconstruction quality is reported with three complementary metrics:
MSE (training loss), PSNR (log-scaled MSE, in dB), and SSIM (structural
similarity, Wang et al. 2004 -- see the `ssim` function), since two
reconstructions can tie on MSE/PSNR while differing a lot structurally
(e.g. blur vs. noise).

Local usage (lightweight smoke test only, no dataset download):
    python code/stl10_conv_autoencoder.py --smoke-test

Colab usage (real data, full training). STL-10 is a much larger
download (~2.6GB) than MNIST/CIFAR-10, so this must run on Colab with
Drive caching, not locally:
    python code/stl10_conv_autoencoder.py --data-root /content/data \
        --epochs 15 --batch-size 64 --lr 1e-3

Colab usage (compression-ratio comparison): trains one model per
latent_dim in --latent-dims from scratch and compares their
reconstructions on a couple of real test images (see
run_compression_ratio_comparison / visualize_compression_comparison):
    python code/stl10_conv_autoencoder.py --data-root /content/data \
        --compare-compression --latent-dims 16,64,256 --compression-epochs 5
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class STL10ConvAutoencoder(nn.Module):
    """4 strided conv blocks (encoder) + FC bottleneck + 4 transposed-conv
    blocks (decoder), every op named explicitly.

    Shape trace for input [B, 3, 96, 96] with latent_dim=128:

        --- encoder ---
        conv1  (3  ->32 , k3, s2, p1) -> [B,  32, 48, 48]
        relu1                          -> [B,  32, 48, 48]
        conv2  (32 ->64 , k3, s2, p1) -> [B,  64, 24, 24]
        relu2                          -> [B,  64, 24, 24]
        conv3  (64 ->128, k3, s2, p1) -> [B, 128, 12, 12]
        relu3                          -> [B, 128, 12, 12]
        conv4  (128->256, k3, s2, p1) -> [B, 256,  6,  6]
        relu4                          -> [B, 256,  6,  6]
        flatten                        -> [B, 9216]
        fc_enc (9216->128)             -> [B,  128]          (latent z)

        --- decoder ---
        fc_dec  (128->9216)            -> [B, 9216]
        relu_dec                       -> [B, 9216]
        unflatten                      -> [B, 256,  6,  6]
        deconv1 (256->128, k3,s2,p1,op1) -> [B, 128, 12, 12]
        relu_d1                          -> [B, 128, 12, 12]
        deconv2 (128->64 , k3,s2,p1,op1) -> [B,  64, 24, 24]
        relu_d2                          -> [B,  64, 24, 24]
        deconv3 (64 ->32 , k3,s2,p1,op1) -> [B,  32, 48, 48]
        relu_d3                          -> [B,  32, 48, 48]
        deconv4 (32 ->3  , k3,s2,p1,op1) -> [B,   3, 96, 96]
        sigmoid                          -> [B,   3, 96, 96]   (x_hat in [0,1])

    Stride-2 3x3 convolutions with padding=1 halve H,W at each encoder
    stage, following out = floor((in + 2*pad - kernel) / stride) + 1.
    96 is divisible by 16 (2^4), so all four halvings land on even
    numbers (96->48->24->12->6) with no rounding ambiguity -- unlike
    the MNIST autoencoder (28->14->7->4), where 7 is odd. Because of
    that, every decoder stage here uses the same output_padding=1
    (whereas MNIST's decoder needed a mix of 0 and 1): for an exact
    power-of-two halving, ConvTranspose2d's
        out = (in-1)*stride - 2*pad + kernel + output_padding
    always needs output_padding=1 to recover out = 2*in exactly, e.g.
    deconv1: in=6 -> (6-1)*2-2*1+3+1 = 10-2+3+1 = 12 (matches encoder's 12).
    A final sigmoid squashes the output to [0,1], matching the range of
    ToTensor()-normalized STL-10 pixels.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # --- encoder: 96->48->24->12->6, channels 3->32->64->128->256 ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()

        # --- bottleneck: [B,256,6,6] <-> [B,9216] <-> [B,latent_dim] ---
        self.flatten = nn.Flatten()  # [B,256,6,6] -> [B, 256*6*6] = [B,9216]
        self.fc_enc = nn.Linear(in_features=256 * 6 * 6, out_features=latent_dim)
        self.fc_dec = nn.Linear(in_features=latent_dim, out_features=256 * 6 * 6)
        self.relu_dec = nn.ReLU()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 6, 6))  # [B,9216] -> [B,256,6,6]

        # --- decoder: 6->12->24->48->96, channels 256->128->64->32->3 ---
        # output_padding=1 at every stage recovers an exact *2 upsampling
        # (see class docstring: 96 is divisible by 16, so every encoder
        # halving is exact, unlike MNIST's odd intermediate size of 7).
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_d1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_d2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_d3 = nn.ReLU()
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """x: [B, 3, 96, 96] in [0,1] -> x_hat: [B, 3, 96, 96] in [0,1]."""

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
        h = step(self.conv4, h, "conv4")
        h = step(self.relu4, h, "relu4")

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
        h = step(self.relu_d3, h, "relu_d3")
        h = step(self.deconv4, h, "deconv4")
        x_hat = step(self.sigmoid, h, "sigmoid")

        return x_hat


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio, PSNR = 10*log10(MAX^2 / MSE), in dB.

    mse: any-shape tensor of mean-squared-errors (scalar or per-image).
    max_val=1.0 matches the [0,1] pixel range used throughout this
    script (ToTensor() input, Sigmoid() output). Higher PSNR = closer
    reconstruction. `eps` avoids log(0) when mse==0 (exact reconstruction).
    This is an exact, deterministic function of MSE, not a separate
    measurement -- it only rescales MSE onto a log scale that is the
    standard way reconstruction quality is reported in the denoising/
    compression literature.
    """
    return 10.0 * torch.log10((max_val ** 2) / (mse + eps))


def compression_ratio(latent_dim: int, image_shape: tuple[int, int, int] = (3, 96, 96)) -> float:
    """input pixel count / latent_dim, e.g. 27648/64 = 432 means the
    bottleneck stores 432x fewer numbers than the raw image."""
    input_dim = image_shape[0] * image_shape[1] * image_shape[2]
    return input_dim / latent_dim


def _gaussian_ssim_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    """Build the depthwise Gaussian window used by `ssim`.

    coords: [window_size], offsets from the window center.
    g_1d:   [window_size], 1D Gaussian weights, sum to 1.
    g_2d = outer(g_1d, g_1d): [window_size, window_size], separable 2D Gaussian.
    window: [channels, 1, window_size, window_size] -- the same 2D
    Gaussian repeated once per channel. F.conv2d(..., groups=channels)
    then convolves each channel with its own copy of the kernel
    independently (a "depthwise" convolution), instead of mixing
    channels together the way a normal Conv2d with groups=1 would --
    SSIM's local statistics (mean/variance/covariance) must stay
    per-channel, not blended across R,G,B.
    """
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2  # [window_size]
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_1d = g_1d / g_1d.sum()  # [window_size], sums to 1 (normalized weighting, like a local average)

    g_2d = g_1d.unsqueeze(1) @ g_1d.unsqueeze(0)  # [window_size,1] @ [1,window_size] -> [window_size,window_size]
    window = g_2d.expand(channels, 1, window_size, window_size).contiguous()  # [C,1,win,win]
    return window


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> torch.Tensor:
    """Structural Similarity Index (Wang et al., 2004), single-scale,
    11x11 Gaussian window (sigma=1.5) -- the paper's standard settings,
    matching skimage's/pytorch-msssim's defaults.

        SSIM(x,y) = [(2*mu_x*mu_y + C1)*(2*sigma_xy + C2)]
                    / [(mu_x^2+mu_y^2+C1)*(sigma_x^2+sigma_y^2+C2)]

    mu_x, mu_y: local (window-weighted) means of x, y.
    sigma_x^2, sigma_y^2: local variances. sigma_xy: local covariance.
    C1=(0.01*max_val)^2, C2=(0.03*max_val)^2: small constants (from the
    paper) that stabilize the division when means/variances are near 0.

    x, y: [B,C,H,W], same shape, values in [0, max_val] (our images are
    in [0,1], so max_val=1.0 matches the Sigmoid-bounded reconstruction
    and ToTensor()-normalized original).

    Local statistics are computed by convolving with the Gaussian
    window (mu = E[x] under the window; sigma_xy = E[xy] - E[x]E[y]),
    which keeps H,W unchanged via padding=window_size//2 (same-size
    convolution), giving a per-pixel SSIM map of shape [B,C,H,W]; we
    then average over C,H,W to get one SSIM score per image, [B].

    SSIM=1 only for x==y exactly; it decreases towards -1 as x,y become
    more dissimilar in luminance/contrast/structure. This is an exact,
    standard implementation of the single-scale SSIM formula (not
    multi-scale MS-SSIM), verified in run_smoke_test by checking
    SSIM(x,x)=1 and that SSIM decreases monotonically as noise grows --
    not cross-checked bit-for-bit against a reference library.
    """
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

    return ssim_map.mean(dim=(1, 2, 3))  # [B], averaged over channels and space -> one score per image


def get_stl10_dataloaders(data_root: str, batch_size: int):
    """Real STL-10 train/test split (torchvision's native partition,
    the labeled 'train' and 'test' splits -- the separate 100k-image
    'unlabeled' split is not used here since labels aren't needed for
    an autoencoder but the labeled splits already give a clean, well-
    established train/test partition).

    Downloads ~2.6GB on first call. Intended for Colab, not local
    smoke tests.
    """
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
    model: STL10ConvAutoencoder,
    images: torch.Tensor,
    num_examples: int = 5,
    save_path: str | None = None,
):
    """Qualitative + quantitative check: top row = original images,
    bottom row = the model's reconstruction, with the per-image
    reconstruction MSE, PSNR, and SSIM (all computed against that
    original image) in each bottom title.

    images: [N, 3, 96, 96] float in [0,1] (ToTensor output), already on
    the same device as `model`.
    """
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
        # RGB images: unlike single-channel MNIST, matplotlib ignores
        # vmin/vmax for 3-channel imshow data, so we rely directly on
        # the [0,1] range already guaranteed by ToTensor() (original)
        # and Sigmoid() (reconstruction).
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
        images = images.to(device)  # [B,3,96,96]; label unused, AE target is the image itself

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


def train_model_for_latent_dim(latent_dim, train_loader, test_loader, device, epochs, lr=1e-3, seed=0):
    """Build and train one STL10ConvAutoencoder at a given bottleneck
    size, from scratch. Returns (trained_model, final_test_mse).

    Each latent_dim defines a differently-shaped `fc_enc`/`fc_dec`
    (in_features/out_features = latent_dim), so comparing compression
    ratios requires training one full model per latent_dim -- there is
    no way to share weights between them.
    """
    torch.manual_seed(seed)
    model = STL10ConvAutoencoder(latent_dim=latent_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    test_mse = float("nan")
    for epoch in range(epochs):
        train_mse = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_mse = evaluate(model, test_loader, loss_fn, device)
        print(
            f"[latent_dim={latent_dim:>4}] epoch {epoch + 1:3d}/{epochs}  "
            f"train_mse={train_mse:.4f}  test_mse={test_mse:.4f}"
        )

    return model, test_mse


def visualize_compression_comparison(
    models_by_latent_dim: dict,
    images: torch.Tensor,
    num_examples: int = 2,
    save_path: str | None = None,
):
    """Side-by-side comparison of reconstruction quality vs. bottleneck
    size: one row per example image, columns = [original, recon at each
    latent_dim in ascending order]. Per-reconstruction MSE, PSNR, and
    compression ratio (input pixels / latent_dim) are shown in each title.

    models_by_latent_dim: {latent_dim: trained STL10ConvAutoencoder},
    all already on the same device as `images`. images: [N,3,96,96]
    float in [0,1], on that same device.
    """
    num_examples = min(num_examples, images.size(0))
    latent_dims = sorted(models_by_latent_dim.keys())
    n_cols = 1 + len(latent_dims)

    originals = images[:num_examples]  # [num_examples, 3, 96, 96]
    image_shape = tuple(originals.shape[1:])  # (3, 96, 96)

    reconstructions_by_dim = {}
    for latent_dim, model in models_by_latent_dim.items():
        model.eval()
        with torch.no_grad():
            reconstructions_by_dim[latent_dim] = model(originals).cpu()  # [num_examples,3,96,96]

    originals_cpu = originals.cpu()

    fig, axes = plt.subplots(num_examples, n_cols, figsize=(2.3 * n_cols, 2.6 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, n_cols)

    for row in range(num_examples):
        original_hwc = originals_cpu[row].permute(1, 2, 0).numpy()  # [3,96,96] -> [96,96,3]
        axes[row, 0].imshow(original_hwc)
        axes[row, 0].set_title("original" if row == 0 else "", fontsize=9)
        axes[row, 0].axis("off")

        for col, latent_dim in enumerate(latent_dims, start=1):
            reconstruction = reconstructions_by_dim[latent_dim][row]  # [3,96,96]
            mse = ((reconstruction - originals_cpu[row]) ** 2).mean()
            psnr = psnr_from_mse(mse).item()
            # ssim() expects batched [B,C,H,W]; unsqueeze(0) adds a size-1 batch dim for this one image
            ssim_val = ssim(reconstruction.unsqueeze(0), originals_cpu[row].unsqueeze(0)).item()
            ratio = compression_ratio(latent_dim, image_shape)

            reconstruction_hwc = reconstruction.permute(1, 2, 0).numpy()
            axes[row, col].imshow(reconstruction_hwc)
            metrics_line = f"MSE={mse.item():.4f} PSNR={psnr:.1f}dB\nSSIM={ssim_val:.3f}"
            title = f"z={latent_dim} ({ratio:.0f}x)\n{metrics_line}" if row == 0 else metrics_line
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis("off")

    fig.suptitle("Reconstruction quality vs. compression ratio (bottleneck size)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved compression-ratio comparison figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def run_compression_ratio_comparison(
    data_root: str,
    batch_size: int = 64,
    epochs: int = 5,
    latent_dims=(16, 64, 256),
    num_example_images: int = 2,
    device: torch.device | None = None,
    seed: int = 0,
):
    """Train one STL10ConvAutoencoder per latent_dim in `latent_dims`
    (each from scratch, `epochs` epochs), then compare their
    reconstructions on `num_example_images` real test images.

    This trains len(latent_dims) separate models -- real GPU training,
    intended for Colab only, not local smoke testing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_stl10_dataloaders(data_root, batch_size)

    models_by_latent_dim = {}
    summary = []
    for latent_dim in latent_dims:
        model, test_mse = train_model_for_latent_dim(
            latent_dim, train_loader, test_loader, device, epochs, seed=seed,
        )
        test_ssim = evaluate_ssim(model, test_loader, device)
        models_by_latent_dim[latent_dim] = model
        summary.append({
            "latent_dim": latent_dim,
            "compression_ratio": compression_ratio(latent_dim),
            "test_mse": test_mse,
            "test_psnr_db": psnr_from_mse(torch.tensor(test_mse)).item(),
            "test_ssim": test_ssim,
        })

    print("\ncompression-ratio comparison summary:")
    for row in summary:
        print(
            f"  latent_dim={row['latent_dim']:>4}  ratio={row['compression_ratio']:>6.0f}x  "
            f"test_mse={row['test_mse']:.4f}  test_psnr={row['test_psnr_db']:.1f}dB  "
            f"test_ssim={row['test_ssim']:.3f}"
        )

    sample_images, _sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)
    visualize_compression_comparison(
        models_by_latent_dim, sample_images, num_examples=num_example_images,
        save_path="stl10_compression_ratio_comparison.png",
    )

    return models_by_latent_dim, summary


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, no real training. Confirms construction, the
    full forward shape trace, parameter count, one backward pass, one
    optimizer step, and the reconstruction-plotting code path.
    """
    torch.manual_seed(0)

    model = STL10ConvAutoencoder(latent_dim=128).to(device)
    n_params = count_trainable_parameters(model)
    print(f"trainable parameters: {n_params:,}")

    dummy_images = torch.rand(4, 3, 96, 96, device=device)  # [B,3,96,96], in [0,1] like real STL-10

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
    print(f"SSIM(x, x+large noise) = {ssim_large_noise:.4f}  (lower, as expected: more noise -> less structural similarity)")

    scratch_png = "stl10_autoencoder_smoke_reconstructions.png"
    visualize_reconstructions(model, dummy_images, num_examples=4, save_path=scratch_png)

    print("\n--- compression-ratio comparison code path (synthetic data, tiny models/epochs) ---")
    from torch.utils.data import DataLoader, TensorDataset

    synthetic_images = torch.rand(8, 3, 96, 96, device=device)  # [8,3,96,96], in [0,1]
    synthetic_labels = torch.zeros(8, dtype=torch.long, device=device)  # unused by the AE, placeholder
    synthetic_loader = DataLoader(TensorDataset(synthetic_images, synthetic_labels), batch_size=4)

    tiny_latent_dims = (8, 32)
    models_by_latent_dim = {}
    for latent_dim in tiny_latent_dims:
        tiny_model, tiny_test_mse = train_model_for_latent_dim(
            latent_dim, synthetic_loader, synthetic_loader, device, epochs=1,
        )
        assert torch.isfinite(torch.tensor(tiny_test_mse)), f"latent_dim={latent_dim} test_mse not finite"
        models_by_latent_dim[latent_dim] = tiny_model

    compression_png = "stl10_compression_comparison_smoke.png"
    visualize_compression_comparison(
        models_by_latent_dim, synthetic_images, num_examples=2, save_path=compression_png,
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
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-compression", action="store_true",
                         help="train one model per --latent-dims value and compare reconstructions "
                              "instead of running the normal single-model training loop")
    parser.add_argument("--latent-dims", type=str, default="16,64,256",
                         help="comma-separated latent_dim values used by --compare-compression")
    parser.add_argument("--compression-epochs", type=int, default=5,
                         help="epochs per model when running --compare-compression")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    if args.compare_compression:
        latent_dims = tuple(int(d) for d in args.latent_dims.split(","))
        run_compression_ratio_comparison(
            args.data_root, batch_size=args.batch_size, epochs=args.compression_epochs,
            latent_dims=latent_dims, device=device, seed=args.seed,
        )
        return

    torch.manual_seed(args.seed)

    model = STL10ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    train_loader, test_loader = get_stl10_dataloaders(args.data_root, args.batch_size)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss = evaluate(model, test_loader, loss_fn, device)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_mse={train_loss:.4f}  test_mse={test_loss:.4f}")

    sample_images, _sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)
    visualize_reconstructions(
        model, sample_images, num_examples=5,
        save_path="stl10_autoencoder_reconstructions.png",
    )


if __name__ == "__main__":
    main()
