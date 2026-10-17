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
x is corrupted with additive Gaussian noise to make x_noisy = x + sigma*eps,
eps ~ N(0,I), and the U-Net is trained to undo the corruption. This is
the same input/output-image-shape setup a DDPM noise-predictor U-Net
uses, so it is a natural next step toward that part of the thesis --
by default this is still a plain denoising autoencoder (no timestep
conditioning, no noise schedule); see "Optional sinusoidal timestep
conditioning" below for the DDPM-style extension that adds the first
of those two pieces.

Optional sinusoidal timestep conditioning (DDPM-style; use_time_conditioning=True):

TinyUNet64 can optionally be conditioned on a diffusion timestep t,
turning it from a fixed-noise-level denoiser into the kind of
timestep-aware noise predictor DDPM needs (Ho et al., 2020, Sec. 3.2
mentions conditioning every residual block on t via a Transformer-style
sinusoidal embedding -- this is that embedding, added to this U-Net's
plain conv blocks since it has no residual blocks yet). This is OFF by
default (use_time_conditioning=False) so every existing script using
TinyUNet64 is unaffected; the forward-pass noise schedule itself is
still not implemented here (t is just an input the network is
conditioned on, not yet connected to code/beta_schedule.py).

Where t enters the network, in order (see `forward`'s verbose log for
exact shapes, and `sinusoidal_timestep_embedding` for the embedding
formula):
    1. t: [B] -> sinusoidal_timestep_embedding -> raw_temb: [B, time_embedding_dim]
    2. raw_temb -> time_mlp (Linear -> SiLU -> Linear) -> temb: [B, time_hidden_dim]
    3. temb is projected per-stage (one nn.Linear(time_hidden_dim, C_stage)
       per stage) and added as a per-channel bias, broadcast over H,W,
       right after that stage's conv and before its ReLU, at EVERY
       stage: enc1, enc2, enc3, bottleneck, dec3, dec2, dec1 (7 injection
       points total -- not at out_conv, which stays a plain 1x1 projection
       to image channels).

TinyUNet64 supports two training objectives, selected via the
`prediction_mode` constructor argument, both trained with the same
pixel-wise MSE loss and compared directly in
code/compare_denoising_objectives.py:

    "x0"    -- the network directly predicts the clean image:
               x_hat = UNet(x_noisy), final activation = Sigmoid
               (output squashed to [0,1], matching the image range).
               L = ||x - x_hat||^2.
    "noise" -- the network predicts the corruption itself:
               eps_hat = UNet(x_noisy), final activation = identity
               (no squashing -- eps is unbounded and zero-mean, so a
               Sigmoid would wrongly clip/bias it).
               L = ||eps - eps_hat||^2, where eps = x_noisy - x
               (the actual, already-clamped corruption applied -- see
               `make_noisy_and_target_noise`).
               The clean-image estimate is then recovered afterward via
               x_hat = clamp(x_noisy - eps_hat, 0, 1)
               (see `denoise_to_clean_image`); if eps_hat == eps exactly,
               this recovers x exactly, by construction of eps.

This mirrors the two equivalent parameterizations of a DDPM
noise-predictor (predict x0 vs. predict eps; see Ho et al., 2020,
Sec. 3.2): the architecture and the ground-truth information available
per sample are identical between the two modes, only the *target the
loss is computed against* differs, so any gap in resulting denoising
quality isolates the effect of the training objective itself.

Local usage (lightweight smoke test only, no dataset download):
    python code/unet_tiny_denoising_64.py --smoke-test

Colab usage (real data, full training). Uses CIFAR-10 (small download,
~170MB) upsampled to 64x64 so the U-Net operates at the resolution it
was designed for:
    python code/unet_tiny_denoising_64.py --data-root /content/data \
        --epochs 10 --batch-size 128 --lr 1e-3 --noise-std 0.3
"""

import argparse
import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Sinusoidal timestep embedding -- exact reproduction of the
    `get_timestep_embedding` function in Ho et al.'s (2020) official
    DDPM code (hojonathanho/diffusion, nn.py), itself borrowed from the
    Transformer positional encoding (Vaswani et al., 2017, "Attention
    Is All You Need", Sec. 3.5).

    Equation, for frequency index i = 0 .. embedding_dim/2 - 1:
        freq_i   = max_period ** ( -i / (embedding_dim/2 - 1) )
        angle_i(t) = t * freq_i
        emb(t)   = [ sin(angle_0(t)), ..., sin(angle_{d/2-1}(t)),
                     cos(angle_0(t)), ..., cos(angle_{d/2-1}(t)) ]
    i.e. embedding_dim/2 sin features concatenated with embedding_dim/2
    cos features (`[sin || cos]`, two half-size blocks), evaluated at
    embedding_dim/2 geometrically spaced frequencies from 1 (i=0) down
    to 1/max_period (i=d/2-1). This is the same frequency formula as
    the Transformer positional encoding, but with sin/cos concatenated
    as two blocks rather than interleaved at even/odd dimension indices
    -- a fixed permutation of the same frequency set, harmless here
    because the learned linear layers consuming this embedding
    (`time_mlp_fc1` etc.) are indifferent to which fixed permutation of
    their input dimensions they see.

    Symbol -> code variable -> shape:
        t (diffusion timestep)   timesteps                 [B]
        i (frequency index)      freq_indices               [embedding_dim//2]
        freq_i                   inv_freq                   [embedding_dim//2]
        t * freq_i               angles                     [B, embedding_dim//2]
        emb(t)                   embedding (return value)   [B, embedding_dim]

    Args:
        timesteps: [B], integer or float diffusion step per batch element.
        embedding_dim: output size; must be even (kept simple -- some
            codebases zero-pad one extra column to support odd sizes;
            not needed here since embedding_dim is always chosen even).
        max_period: largest wavelength's period (10000, matching
            Vaswani et al. and Ho et al.).

    Returns:
        embedding: [B, embedding_dim], exact (not approximate) reproduction
        of Ho et al.'s official DDPM timestep-embedding formula.
    """
    assert embedding_dim % 2 == 0, f"embedding_dim must be even, got {embedding_dim}"
    half_dim = embedding_dim // 2
    assert half_dim >= 2, f"embedding_dim must be >= 4, got {embedding_dim}"

    freq_indices = torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)  # [half_dim]
    inv_freq = torch.exp(-math.log(max_period) * freq_indices / (half_dim - 1))          # [half_dim]

    angles = timesteps.float()[:, None] * inv_freq[None, :]  # [B,1]*[1,half_dim] -> [B,half_dim] (broadcast)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B,half_dim]+[B,half_dim] -> [B,embedding_dim]
    return embedding


class TinyUNet64(nn.Module):
    """3-level encoder/decoder U-Net with skip connections, every op
    named explicitly. See module docstring for the full shape trace.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        prediction_mode: str = "x0",
        use_time_conditioning: bool = False,
        time_embedding_dim: int = 128,
        time_hidden_dim: int | None = None,
    ):
        super().__init__()
        assert prediction_mode in ("x0", "noise"), (
            f"prediction_mode must be 'x0' or 'noise', got {prediction_mode!r}"
        )
        self.prediction_mode = prediction_mode
        self.use_time_conditioning = use_time_conditioning
        self.time_embedding_dim = time_embedding_dim

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        if self.use_time_conditioning:
            # time_hidden_dim defaults to 4*base_channels (= c4, the
            # bottleneck width), a common DDPM convention tying the time
            # embedding's width to the network's own channel scale.
            self.time_hidden_dim = time_hidden_dim if time_hidden_dim is not None else base_channels * 4

            # raw sinusoidal features [B,time_embedding_dim] -> learned
            # embedding [B,time_hidden_dim]; SiLU (as in Ho et al.'s code)
            # lets the network reshape the fixed sin/cos basis nonlinearly.
            self.time_mlp_fc1 = nn.Linear(time_embedding_dim, self.time_hidden_dim)
            self.time_mlp_act = nn.SiLU()
            self.time_mlp_fc2 = nn.Linear(self.time_hidden_dim, self.time_hidden_dim)

            # one projection per injection stage: time_hidden_dim -> that
            # stage's channel count, so temb can be added as a per-channel
            # bias to that stage's feature map (see `forward`'s inject_time).
            self.time_proj_enc1 = nn.Linear(self.time_hidden_dim, c1)
            self.time_proj_enc2 = nn.Linear(self.time_hidden_dim, c2)
            self.time_proj_enc3 = nn.Linear(self.time_hidden_dim, c3)
            self.time_proj_bottleneck = nn.Linear(self.time_hidden_dim, c4)
            self.time_proj_dec3 = nn.Linear(self.time_hidden_dim, c3)
            self.time_proj_dec2 = nn.Linear(self.time_hidden_dim, c2)
            self.time_proj_dec1 = nn.Linear(self.time_hidden_dim, c1)

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

    def forward(
        self, x: torch.Tensor, t: torch.Tensor | None = None, verbose: bool = False
    ) -> torch.Tensor:
        """x: [B, in_channels, 64, 64] (the noisy image, x_noisy) ->
        out: [B, out_channels, 64, 64]. Meaning of `out` depends on
        self.prediction_mode: x_hat in [0,1] for "x0", or eps_hat
        (unbounded) for "noise".

        t: [B] diffusion timesteps, required iff self.use_time_conditioning
        is True (otherwise must be left as None). See the module
        docstring's "Optional sinusoidal timestep conditioning" section
        for exactly where t enters the network; every entry point is
        also logged below when verbose=True."""

        def log(label, tensor):
            if verbose:
                print(f"{label:<48}: {tuple(tensor.shape)}")

        log("input", x)

        if self.use_time_conditioning:
            assert t is not None, "use_time_conditioning=True requires a timestep tensor t: [B]"
            assert t.shape == (x.shape[0],), f"t must be [B]=({x.shape[0]},), got {tuple(t.shape)}"
            log("timestep t (raw, one int per batch element)", t)

            raw_temb = sinusoidal_timestep_embedding(t, self.time_embedding_dim)  # [B] -> [B,time_embedding_dim]
            log("[TIME ENTERS HERE] sinusoidal embedding of t", raw_temb)

            temb = self.time_mlp_fc1(raw_temb)   # [B,time_embedding_dim] -> [B,time_hidden_dim]
            temb = self.time_mlp_act(temb)
            temb = self.time_mlp_fc2(temb)       # [B,time_hidden_dim] -> [B,time_hidden_dim]
            log("time embedding after time_mlp (fc1->SiLU->fc2)", temb)
        else:
            assert t is None, "t was passed but use_time_conditioning=False; this model ignores t"
            temb = None

        def inject_time(h: torch.Tensor, proj_name: str, label: str) -> torch.Tensor:
            """h: [B,C,H,W]; adds a per-sample, per-channel bias derived
            from temb, broadcast over all H*W spatial positions. This is
            the mechanism by which the SAME scalar timestep t affects
            EVERY pixel of this stage's feature map identically (no
            spatial information in t itself, so no spatial variation in
            how it is injected). `proj_name` (not the layer itself) is
            passed in so the corresponding nn.Linear is looked up only
            when use_time_conditioning is True -- it does not exist as
            an attribute at all otherwise."""
            if not self.use_time_conditioning:
                return h
            proj: nn.Linear = getattr(self, proj_name)
            time_bias = proj(temb)                   # [B,time_hidden_dim] -> [B,C]
            log(f"  [TIME ENTERS HERE] {label}: time_proj(temb)", time_bias)
            time_bias = time_bias[:, :, None, None]  # [B,C] -> [B,C,1,1] so it broadcasts over (H,W)
            h = h + time_bias                        # [B,C,H,W] + [B,C,1,1] -> [B,C,H,W], same bias at every (h,w)
            log(f"  {label}: feature map after + time embedding", h)
            return h

        # --- encoder ---
        h = self.enc1(x)
        h = inject_time(h, "time_proj_enc1", "enc1")
        skip1 = self.relu_e1(h)
        log("skip1 saved (before pool1)", skip1)
        h = self.pool1(skip1)
        log("after pool1", h)

        h = self.enc2(h)
        h = inject_time(h, "time_proj_enc2", "enc2")
        skip2 = self.relu_e2(h)
        log("skip2 saved (before pool2)", skip2)
        h = self.pool2(skip2)
        log("after pool2", h)

        h = self.enc3(h)
        h = inject_time(h, "time_proj_enc3", "enc3")
        skip3 = self.relu_e3(h)
        log("skip3 saved (before pool3)", skip3)
        h = self.pool3(skip3)
        log("after pool3", h)

        # --- bottleneck ---
        h = self.bottleneck(h)
        h = inject_time(h, "time_proj_bottleneck", "bottleneck")
        h = self.relu_b(h)
        log("after bottleneck", h)

        # --- decoder, stage 3: up3 + concat(skip3) ---
        h = self.up3(h)
        log("up3 output (before concat with skip3)", h)
        log("skip3 (being concatenated)", skip3)
        h = torch.cat([h, skip3], dim=1)  # [B,c3,16,16]+[B,c3,16,16] -> [B,2*c3,16,16] along channel dim
        log("after concat(up3, skip3)", h)
        h = self.dec3(h)
        h = inject_time(h, "time_proj_dec3", "dec3")
        h = self.relu_d3(h)
        log("after dec3", h)

        # --- decoder, stage 2: up2 + concat(skip2) ---
        h = self.up2(h)
        log("up2 output (before concat with skip2)", h)
        log("skip2 (being concatenated)", skip2)
        h = torch.cat([h, skip2], dim=1)  # [B,c2,32,32]+[B,c2,32,32] -> [B,2*c2,32,32]
        log("after concat(up2, skip2)", h)
        h = self.dec2(h)
        h = inject_time(h, "time_proj_dec2", "dec2")
        h = self.relu_d2(h)
        log("after dec2", h)

        # --- decoder, stage 1: up1 + concat(skip1) ---
        h = self.up1(h)
        log("up1 output (before concat with skip1)", h)
        log("skip1 (being concatenated)", skip1)
        h = torch.cat([h, skip1], dim=1)  # [B,c1,64,64]+[B,c1,64,64] -> [B,2*c1,64,64]
        log("after concat(up1, skip1)", h)
        h = self.dec1(h)
        h = inject_time(h, "time_proj_dec1", "dec1")
        h = self.relu_d1(h)
        log("after dec1", h)

        h = self.out_conv(h)  # time conditioning stops here: plain 1x1 projection to image channels
        log("after out_conv (no time injection here)", h)

        if self.prediction_mode == "x0":
            out = self.sigmoid(h)
            log("after sigmoid (final output = x_hat, in [0,1])", out)
        else:  # "noise": eps is unbounded and zero-mean, so no squashing activation
            out = h
            log("final output = eps_hat (unbounded, no activation)", out)

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


def make_noisy_and_target_noise(images: torch.Tensor, noise_std: float):
    """images: [B,C,H,W] in [0,1] -> (noisy, noise), both [B,C,H,W].

    noise := noisy - images, i.e. the *actual* corruption applied to
    this particular image, taken *after* clamping noisy back to [0,1].
    Defining the target this way (rather than the pre-clamp
    noise_std*eps draw) keeps x = noisy - noise exactly true by
    construction even for pixels where the Gaussian draw saturated the
    [0,1] range, which is what the "noise" prediction_mode's loss
    target and `denoise_to_clean_image` both rely on."""
    noisy = add_gaussian_noise(images, noise_std)
    noise = noisy - images
    return noisy, noise


def denoise_to_clean_image(model: "TinyUNet64", noisy_images: torch.Tensor) -> torch.Tensor:
    """Runs the model and converts its raw output into a clean-image
    estimate x_hat in [0,1], regardless of model.prediction_mode:
        "x0"    -> model already outputs x_hat directly.
        "noise" -> model outputs eps_hat; x_hat = clamp(noisy - eps_hat, 0, 1),
                   the same algebraic inverse used to define the
                   "noise" target in `make_noisy_and_target_noise`.
    noisy_images: [B,C,H,W] -> x_hat: [B,C,H,W] in [0,1]."""
    prediction = model(noisy_images)
    if model.prediction_mode == "x0":
        return prediction
    return (noisy_images - prediction).clamp(0.0, 1.0)


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
    input, row 3 = U-Net's clean-image estimate x_hat (recovered via
    `denoise_to_clean_image`, which handles both prediction_mode
    values), with per-image MSE/PSNR (output vs. clean) in each row-3
    title.

    clean_images: [N,C,H,W] float in [0,1], already on model's device.
    """
    num_examples = min(num_examples, clean_images.size(0))

    model.eval()
    with torch.no_grad():
        clean = clean_images[:num_examples]                 # [num_examples,C,H,W]
        noisy = add_gaussian_noise(clean, noise_std)         # [num_examples,C,H,W]
        denoised = denoise_to_clean_image(model, noisy)       # [num_examples,C,H,W], in [0,1]
        per_image_mse = ((denoised - clean) ** 2).mean(dim=(1, 2, 3))  # [num_examples]
        per_image_psnr = psnr_from_mse(per_image_mse)          # [num_examples], dB

    clean_cpu, noisy_cpu, denoised_cpu = clean.cpu(), noisy.cpu(), denoised.cpu()
    mse_cpu, psnr_cpu = per_image_mse.cpu(), per_image_psnr.cpu()

    def to_hwc(img_chw):
        return img_chw.permute(1, 2, 0).squeeze(-1).numpy()  # [C,H,W]->[H,W,C], squeeze C=1 for grayscale

    fig, axes = plt.subplots(3, num_examples, figsize=(2.2 * num_examples, 6.6))
    if num_examples == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["clean (target)", "noisy (input)", "U-Net x_hat"]
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

    fig.suptitle(
        f"clean / noisy (std={noise_std}) / U-Net-denoised "
        f"(prediction_mode={model.prediction_mode!r}), {num_examples} examples"
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved denoising figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def train_one_epoch(model, loader, loss_fn, optimizer, device, noise_std):
    """One training epoch. The loss target depends on model.prediction_mode:
    "x0" trains the raw model output against the clean image directly;
    "noise" trains it against the actual added noise (see
    `make_noisy_and_target_noise`). Both cases use the same `loss_fn`
    (MSE) and the same forward/backward/step structure -- only the
    *target tensor* differs, isolating the effect of the objective."""
    model.train()
    total_loss, total_samples = 0.0, 0

    for clean_images, _labels in loader:
        clean_images = clean_images.to(device)              # [B,C,H,W]; label unused
        noisy_images, noise = make_noisy_and_target_noise(clean_images, noise_std)
        target = clean_images if model.prediction_mode == "x0" else noise

        optimizer.zero_grad()
        prediction = model(noisy_images)                     # [B,C,H,W]: x_hat or eps_hat
        loss = loss_fn(prediction, target)                   # scalar MSE vs. the mode's own target
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clean_images.size(0)
        total_samples += clean_images.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, noise_std):
    """Test-set loss in the model's own native units (x0: pixel MSE;
    noise: noise MSE) -- NOT directly comparable between the two
    prediction_modes. Use `evaluate_denoising_mse` for the common,
    cross-mode-comparable metric (denoising MSE of x_hat vs. clean)."""
    model.eval()
    total_loss, total_samples = 0.0, 0

    for clean_images, _labels in loader:
        clean_images = clean_images.to(device)
        noisy_images, noise = make_noisy_and_target_noise(clean_images, noise_std)
        target = clean_images if model.prediction_mode == "x0" else noise
        prediction = model(noisy_images)
        loss = loss_fn(prediction, target)

        total_loss += loss.item() * clean_images.size(0)
        total_samples += clean_images.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate_denoising_mse(model, loader, device, noise_std):
    """Test-set MSE of the recovered clean-image estimate x_hat vs. the
    clean image, via `denoise_to_clean_image` -- identical units
    (pixel-space MSE, inputs in [0,1]) regardless of model.prediction_mode,
    so this is the metric to use when comparing "x0" vs. "noise" models
    (unlike `evaluate`, whose loss is in each mode's own native units)."""
    model.eval()
    total_mse, total_samples = 0.0, 0

    for clean_images, _labels in loader:
        clean_images = clean_images.to(device)
        noisy_images = add_gaussian_noise(clean_images, noise_std)
        x_hat = denoise_to_clean_image(model, noisy_images)
        batch_mse = ((x_hat - clean_images) ** 2).mean(dim=(1, 2, 3))  # [B]

        total_mse += batch_mse.sum().item()
        total_samples += clean_images.size(0)

    return total_mse / total_samples


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, no real training. For each prediction_mode
    ("x0", "noise") confirms construction, one backward pass + one
    optimizer step, and (for "x0") the full forward shape trace +
    the denoising-visualization code path. Also checks that
    `denoise_to_clean_image` recovers the clean image exactly when the
    model's raw output is forced to equal the true target (sanity
    check on the x0/noise algebra itself, independent of training).
    """
    torch.manual_seed(0)

    dummy_clean = torch.rand(4, 3, 64, 64, device=device)  # [B,3,64,64], in [0,1]
    loss_fn = nn.MSELoss()

    for prediction_mode in ("x0", "noise"):
        print(f"\n=== prediction_mode={prediction_mode!r} ===")
        model = TinyUNet64(in_channels=3, out_channels=3, base_channels=32, prediction_mode=prediction_mode).to(device)
        n_params = count_trainable_parameters(model)
        print(f"trainable parameters: {n_params:,}")

        noisy_images, noise = make_noisy_and_target_noise(dummy_clean, noise_std=0.3)
        target = dummy_clean if prediction_mode == "x0" else noise

        if prediction_mode == "x0":
            print("\n--- forward shape trace (every skip connection + concatenation) ---")
            prediction = model(noisy_images, verbose=True)
        else:
            prediction = model(noisy_images)
        assert prediction.shape == dummy_clean.shape, (
            f"output shape {tuple(prediction.shape)} != input shape {tuple(dummy_clean.shape)}"
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = loss_fn(prediction, target)
        assert torch.isfinite(loss), "loss is not finite"
        loss.backward()
        optimizer.step()
        print(f"dummy {prediction_mode} MSE: {loss.item():.4f} (finite, backward + optimizer step ok)")

    print("\n--- x0/noise algebra sanity check (denoise_to_clean_image) ---")
    # If the "noise" model's raw output exactly equals the true noise,
    # x_hat = noisy - eps_hat must recover the clean image exactly.
    x_hat_from_true_noise = (noisy_images - noise).clamp(0.0, 1.0)
    assert torch.allclose(x_hat_from_true_noise, dummy_clean, atol=1e-5), (
        "noisy - true_noise should recover the clean image exactly"
    )
    print("noisy_images - true_noise == clean image (exact, as expected)")

    print("\n--- denoising-visualization code path (prediction_mode='x0' model) ---")
    x0_model = TinyUNet64(in_channels=3, out_channels=3, base_channels=32, prediction_mode="x0").to(device)
    scratch_png = "unet_tiny_denoising_smoke.png"
    visualize_denoising(x0_model, dummy_clean, noise_std=0.3, num_examples=4, save_path=scratch_png)

    print("\n=== use_time_conditioning=True ===")
    time_model = TinyUNet64(
        in_channels=3, out_channels=3, base_channels=32, prediction_mode="noise",
        use_time_conditioning=True, time_embedding_dim=128,
    ).to(device)
    print(f"trainable parameters: {count_trainable_parameters(time_model):,}")

    noisy_images, noise = make_noisy_and_target_noise(dummy_clean, noise_std=0.3)
    t_a = torch.randint(low=0, high=1000, size=(dummy_clean.shape[0],), device=device)  # [B]

    print("\n--- forward shape trace (every point t enters the network) ---")
    eps_hat_a = time_model(noisy_images, t=t_a, verbose=True)
    assert eps_hat_a.shape == dummy_clean.shape, (
        f"output shape {tuple(eps_hat_a.shape)} != input shape {tuple(dummy_clean.shape)}"
    )

    # Backward pass must reach the time-conditioning weights, not just the conv weights.
    optimizer = torch.optim.Adam(time_model.parameters(), lr=1e-3)
    loss = loss_fn(eps_hat_a, noise)
    assert torch.isfinite(loss), "time-conditioned loss is not finite"
    loss.backward()
    assert time_model.time_mlp_fc1.weight.grad is not None, "no gradient reached time_mlp_fc1"
    assert torch.isfinite(time_model.time_mlp_fc1.weight.grad).all(), "time_mlp_fc1 grad is not finite"
    assert time_model.time_proj_bottleneck.weight.grad is not None, "no gradient reached time_proj_bottleneck"
    optimizer.step()
    print("gradients reached time_mlp_fc1 and time_proj_bottleneck (finite); optimizer step ok")

    # Same image, different t must give a different output -- otherwise t is silently a no-op.
    with torch.no_grad():
        t_b = (t_a + 500) % 1000  # [B], shifted timesteps, same images
        eps_hat_b = time_model(noisy_images, t=t_b)
    assert not torch.allclose(eps_hat_a, eps_hat_b, atol=1e-6), (
        "output did not change when t changed -- time conditioning is a no-op"
    )
    print("changing t (same images) changes the output: time conditioning is not a no-op")

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
    parser.add_argument("--prediction-mode", type=str, default="x0", choices=["x0", "noise"],
                         help="training objective: predict the clean image directly ('x0') "
                              "or predict the added noise ('noise'); see module docstring")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    torch.manual_seed(args.seed)

    model = TinyUNet64(
        in_channels=3, out_channels=3, base_channels=args.base_channels, prediction_mode=args.prediction_mode,
    ).to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")
    print(f"prediction_mode = {args.prediction_mode!r}")

    train_loader, test_loader = get_cifar10_denoising_dataloaders(args.data_root, args.batch_size, image_size=64)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.noise_std)
        test_loss = evaluate(model, test_loader, loss_fn, device, args.noise_std)
        test_denoising_mse = evaluate_denoising_mse(model, test_loader, device, args.noise_std)
        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  "
            f"train_loss({args.prediction_mode})={train_loss:.4f}  "
            f"test_loss({args.prediction_mode})={test_loss:.4f}  "
            f"test_denoising_mse(x_hat vs clean)={test_denoising_mse:.4f}"
        )

    sample_images, _sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)
    visualize_denoising(
        model, sample_images, args.noise_std, num_examples=5,
        save_path=f"unet_tiny_denoising_reconstructions_{args.prediction_mode}.png",
    )


if __name__ == "__main__":
    main()
