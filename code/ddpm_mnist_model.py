"""Minimal time-conditioned U-Net noise predictor for MNIST (28x28x1
grayscale digits), the "model" piece of a from-scratch DDPM (Ho et al.,
2020, "Denoising Diffusion Probabilistic Models"). No diffusers, no
external model library -- raw torch.nn layers only, every op named
(no nn.Sequential), with a full verbose shape trace in `forward`.

MNIST is used at its NATIVE 28x28 resolution (no padding to 32x32):
28 is exactly divisible by 2 twice (28 -> 14 -> 7), so a 2-downsampling
-stage U-Net needs no padding/cropping tricks to make every skip
connection line up. This keeps the architecture genuinely minimal,
matching this exercise's "minimal DDPM" scope -- a 3rd downsampling
stage (7 -> 3.5) would need padding, which is unnecessary complexity
here. See code/unet_time_conditioned_64.py for the (deeper, 3-stage)
64x64x3 sibling architecture this one borrows its time-conditioning
mechanism from.

Architecture (2 downsampling stages, input [B,1,28,28]):

    --- encoder ---
    enc1 (1 ->c1, k3,p1) -> +time -> relu   -> [B,c1,28,28]  = skip1
    pool1 (2x2 maxpool)                     -> [B,c1,14,14]
    enc2 (c1->c2, k3,p1) -> +time -> relu   -> [B,c2,14,14]  = skip2
    pool2 (2x2 maxpool)                     -> [B,c2, 7, 7]

    --- bottleneck ---
    bottleneck (c2->c3, k3,p1) -> +time -> relu -> [B,c3, 7, 7]

    --- decoder ---
    up2 (ConvTranspose2d c3->c2,k2,s2)      -> [B,c2,14,14]
    concat(up2, skip2)  [c2]+[c2]           -> [B,2*c2,14,14]
    dec2 (2*c2->c2, k3,p1) -> +time -> relu -> [B,c2,14,14]
    up1 (ConvTranspose2d c2->c1,k2,s2)      -> [B,c1,28,28]
    concat(up1, skip1)  [c1]+[c1]           -> [B,2*c1,28,28]
    dec1 (2*c1->c1, k3,p1) -> +time -> relu -> [B,c1,28,28]
    out_conv (c1->1, k1)   (no time here)   -> [B, 1,28,28]

Output is always eps_hat (predicted noise, unbounded, no final
activation) -- the standard DDPM noise-prediction parameterization.

--- How the timestep enters the network ---

    t: [B] (one integer diffusion timestep per batch element)
       | sinusoidal_timestep_embedding(t, time_embedding_dim) -- fixed,
       | no learned parameters (imported from unet_time_conditioned_64.py,
       | exact reproduction of Ho et al.'s official embedding formula)
       v
    raw_temb: [B, time_embedding_dim]
       | time_mlp: Linear -> SiLU -> Linear  -- LEARNED reshaping
       v
    temb: [B, time_hidden_dim]
       | one nn.Linear(time_hidden_dim, C_stage) PER STAGE, reshaped
       | [B,C_stage] -> [B,C_stage,1,1] and ADDED to that stage's
       | feature map (broadcast over H,W), right after that stage's
       | conv and right before its ReLU.
       v
    added into: enc1, enc2, bottleneck, dec2, dec1  (5 injection points;
    NOT out_conv, which stays a plain 1x1 projection to image channels)

Local usage (lightweight smoke test only, no dataset download):
    python code/ddpm_mnist_model.py --smoke-test
"""

import argparse

import torch
import torch.nn as nn

from unet_time_conditioned_64 import sinusoidal_timestep_embedding


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MnistUNet(nn.Module):
    """2-level encoder/decoder U-Net, always conditioned on a diffusion
    timestep t. See module docstring for the full architecture and the
    "How the timestep enters the network" shape trace.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        time_embedding_dim: int = 64,
        time_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        # time_hidden_dim defaults to 4*base_channels (= c3, the
        # bottleneck width), the same convention used in
        # TimeConditionedUNet64.
        self.time_hidden_dim = time_hidden_dim if time_hidden_dim is not None else base_channels * 4

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # --- timestep embedding path ---
        self.time_mlp_fc1 = nn.Linear(time_embedding_dim, self.time_hidden_dim)
        self.time_mlp_act = nn.SiLU()
        self.time_mlp_fc2 = nn.Linear(self.time_hidden_dim, self.time_hidden_dim)

        # One projection per injection stage: time_hidden_dim -> that
        # stage's channel count, added as a per-channel bias (see
        # `inject_time` in forward()). Each width matches the conv
        # stage it feeds, defined below.
        self.time_proj_enc1 = nn.Linear(self.time_hidden_dim, c1)        # matches enc1 output width (c1)
        self.time_proj_enc2 = nn.Linear(self.time_hidden_dim, c2)        # matches enc2 output width (c2)
        self.time_proj_bottleneck = nn.Linear(self.time_hidden_dim, c3)  # matches bottleneck output width (c3)
        self.time_proj_dec2 = nn.Linear(self.time_hidden_dim, c2)        # matches dec2 output width (c2)
        self.time_proj_dec1 = nn.Linear(self.time_hidden_dim, c1)        # matches dec1 output width (c1)

        # --- encoder: 28->14->7, channels in->c1->c2 ---
        self.enc1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.relu_e1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.relu_e2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # --- bottleneck: 7x7, channels c2->c3 ---
        self.bottleneck = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.relu_b = nn.ReLU()

        # --- decoder: 7->14->28, channels c3->c2->c1 ---
        # kernel=stride=2, padding=0 => out = (in-1)*2 + 2 = 2*in exactly,
        # so the upsampled map always matches its skip tensor's H,W.
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1)  # concat(up2, skip2) -> 2*c2 channels in
        self.relu_d2 = nn.ReLU()
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1)  # concat(up1, skip1) -> 2*c1 channels in
        self.relu_d1 = nn.ReLU()

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """x: [B, in_channels, 28, 28] (the noisy image, x_t).
        t: [B] diffusion timesteps, one integer per batch element.
        Returns eps_hat: [B, out_channels, 28, 28], the predicted noise
        (unbounded, no final activation)."""

        def log(label, tensor):
            if verbose:
                print(f"{label:<48}: {tuple(tensor.shape)}")

        assert t.shape == (x.shape[0],), f"t must be [B]=({x.shape[0]},), got {tuple(t.shape)}"
        log("input x", x)
        log("input t (raw, one int per batch element)", t)

        raw_temb = sinusoidal_timestep_embedding(t, self.time_embedding_dim)  # [B] -> [B,time_embedding_dim]
        log("[TIME ENTERS HERE] sinusoidal embedding of t", raw_temb)

        temb = self.time_mlp_fc1(raw_temb)
        temb = self.time_mlp_act(temb)
        temb = self.time_mlp_fc2(temb)       # [B,time_embedding_dim] -> [B,time_hidden_dim]
        log("time embedding after time_mlp (fc1->SiLU->fc2)", temb)

        def inject_time(h: torch.Tensor, proj: nn.Linear, label: str) -> torch.Tensor:
            """h: [B,C,H,W] -> h + proj(temb) broadcast over H,W."""
            time_bias = proj(temb)                   # [B,time_hidden_dim] -> [B,C]
            log(f"  [TIME ENTERS HERE] {label}: time_proj(temb)", time_bias)
            time_bias = time_bias[:, :, None, None]  # [B,C] -> [B,C,1,1]
            h = h + time_bias                        # [B,C,H,W] + [B,C,1,1] -> [B,C,H,W]
            log(f"  {label}: feature map after + time embedding", h)
            return h

        # --- encoder ---
        h = self.enc1(x)
        h = inject_time(h, self.time_proj_enc1, "enc1")
        skip1 = self.relu_e1(h)
        log("skip1 saved (before pool1)", skip1)
        h = self.pool1(skip1)
        log("after pool1", h)

        h = self.enc2(h)
        h = inject_time(h, self.time_proj_enc2, "enc2")
        skip2 = self.relu_e2(h)
        log("skip2 saved (before pool2)", skip2)
        h = self.pool2(skip2)
        log("after pool2", h)

        # --- bottleneck: lowest resolution (7x7), widest channels (c3) ---
        h = self.bottleneck(h)
        h = inject_time(h, self.time_proj_bottleneck, "bottleneck")
        h = self.relu_b(h)
        log("after bottleneck", h)

        # --- decoder, stage 2: upsample -> concat skip2 -> conv -> +time -> relu ---
        h = self.up2(h)
        log("up2 output (before concat with skip2)", h)
        log("skip2 (being concatenated)", skip2)
        h = torch.cat([h, skip2], dim=1)  # [B,c2,14,14]+[B,c2,14,14] -> [B,2*c2,14,14]
        log("after concat(up2, skip2)", h)
        h = self.dec2(h)
        h = inject_time(h, self.time_proj_dec2, "dec2")
        h = self.relu_d2(h)
        log("after dec2", h)

        # --- decoder, stage 1: upsample -> concat skip1 -> conv -> +time -> relu ---
        h = self.up1(h)
        log("up1 output (before concat with skip1)", h)
        log("skip1 (being concatenated)", skip1)
        h = torch.cat([h, skip1], dim=1)  # [B,c1,28,28]+[B,c1,28,28] -> [B,2*c1,28,28]
        log("after concat(up1, skip1)", h)
        h = self.dec1(h)
        h = inject_time(h, self.time_proj_dec1, "dec1")
        h = self.relu_d1(h)
        log("after dec1", h)

        eps_hat = self.out_conv(h)  # time conditioning stops here: plain 1x1 projection to image channels
        log("after out_conv = eps_hat (no time injection, no activation)", eps_hat)
        return eps_hat


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, no
    dataset download, no real training. Confirms construction, the
    full forward shape trace (every point t enters the network),
    one backward pass + one optimizer step reaching the time-embedding
    weights, and that changing t (same images) actually changes the
    output (i.e. t is not a silent no-op)."""
    torch.manual_seed(0)

    model = MnistUNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    dummy_noisy_images = torch.rand(4, 1, 28, 28, device=device)  # [B,1,28,28], in [0,1]
    dummy_true_noise = torch.randn(4, 1, 28, 28, device=device)   # [B,1,28,28], the eps the model should predict
    t_a = torch.randint(low=0, high=1000, size=(4,), device=device)  # [B]

    print("\n--- forward shape trace (every point t enters the network) ---")
    eps_hat_a = model(dummy_noisy_images, t_a, verbose=True)
    assert eps_hat_a.shape == dummy_true_noise.shape, (
        f"output shape {tuple(eps_hat_a.shape)} != target shape {tuple(dummy_true_noise.shape)}"
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = loss_fn(eps_hat_a, dummy_true_noise)
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    assert model.time_mlp_fc1.weight.grad is not None, "no gradient reached time_mlp_fc1"
    assert torch.isfinite(model.time_mlp_fc1.weight.grad).all(), "time_mlp_fc1 grad is not finite"
    assert model.time_proj_bottleneck.weight.grad is not None, "no gradient reached time_proj_bottleneck"
    optimizer.step()
    print(f"\ndummy noise-prediction MSE: {loss.item():.4f} (finite, backward + optimizer step ok)")
    print("gradients reached time_mlp_fc1 and time_proj_bottleneck (finite)")

    with torch.no_grad():
        t_b = (t_a + 500) % 1000  # [B], shifted timesteps, same images
        eps_hat_b = model(dummy_noisy_images, t_b)
    assert not torch.allclose(eps_hat_a, eps_hat_b, atol=1e-6), (
        "output did not change when t changed -- time conditioning is a no-op"
    )
    print("changing t (same images) changes the output: time conditioning is not a no-op")

    print("\nsmoke test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="run lightweight local verification with synthetic tensors, no download")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    raise SystemExit(
        "This file only defines the MNIST DDPM model (use --smoke-test). "
        "See code/ddpm_mnist_scheduler.py for the noise schedule, "
        "code/ddpm_mnist_train.py for the training loop, and "
        "code/ddpm_mnist_sample.py for sampling."
    )


if __name__ == "__main__":
    main()
