"""Time-conditioned educational U-Net for 64x64x3 images: the same
encoder/decoder-with-skip-connections architecture as `TinyUNet64`
(code/unet_tiny_denoising_64.py), but with the diffusion timestep t
ALWAYS part of the forward pass -- no boolean flags, no branching on
"is time conditioning on". This file exists purely so the time-embedding
mechanism can be read top-to-bottom with nothing else mixed in.

Every conv/pool/upsample/concat/time-injection is a named layer (no
nn.Sequential) and forward() prints the [B,C,H,W] (or [B,D]) shape
before AND after every one of those operations when verbose=True.

Architecture (identical skeleton to TinyUNet64, 3 downsampling stages,
input [B,3,64,64]):

    --- encoder ---
    enc1 (3  ->32 , k3,p1) -> +time -> relu   -> [B, 32, 64, 64]   = skip1
    pool1 (2x2 maxpool)                       -> [B, 32, 32, 32]
    enc2 (32 ->64 , k3,p1) -> +time -> relu   -> [B, 64, 32, 32]   = skip2
    pool2 (2x2 maxpool)                       -> [B, 64, 16, 16]
    enc3 (64 ->128, k3,p1) -> +time -> relu   -> [B,128, 16, 16]   = skip3
    pool3 (2x2 maxpool)                       -> [B,128,  8,  8]

    --- bottleneck ---
    bottleneck (128->256, k3,p1) -> +time -> relu -> [B,256,  8,  8]

    --- decoder ---
    up3 (ConvTranspose2d 256->128,k2,s2)      -> [B,128, 16, 16]
    concat(up3, skip3)  [128]+[128]           -> [B,256, 16, 16]
    dec3 (256->128, k3,p1) -> +time -> relu   -> [B,128, 16, 16]
    up2 (ConvTranspose2d 128->64 ,k2,s2)      -> [B, 64, 32, 32]
    concat(up2, skip2)  [64]+[64]             -> [B,128, 32, 32]
    dec2 (128->64 , k3,p1) -> +time -> relu   -> [B, 64, 32, 32]
    up1 (ConvTranspose2d 64 ->32 ,k2,s2)      -> [B, 32, 64, 64]
    concat(up1, skip1)  [32]+[32]             -> [B, 64, 64, 64]
    dec1 (64 ->32 , k3,p1) -> +time -> relu   -> [B, 32, 64, 64]
    out_conv (32->3, k1)   (no time here)     -> [B,  3, 64, 64]

Output is always eps_hat (the predicted noise, unbounded, no final
activation) -- the standard DDPM noise-prediction parameterization that
timestep conditioning is actually for. (TinyUNet64's x0-vs-noise
objective comparison is a separate, orthogonal exercise; it is not
reproduced here to keep this file about ONE thing: where and how t
enters the network.)

--- How the timestep enters the network (the whole point of this file) ---

    t: [B] (one integer timestep per batch element)
       |
       | sinusoidal_timestep_embedding(t, time_embedding_dim)  -- fixed
       | sin/cos "fingerprint" of t, no learned parameters
       v
    raw_temb: [B, time_embedding_dim]
       |
       | time_mlp: Linear -> SiLU -> Linear  -- LEARNED reshaping
       v
    temb: [B, time_hidden_dim]
       |
       | one nn.Linear(time_hidden_dim, C_stage) PER STAGE, applied to
       | temb, reshaped [B,C_stage] -> [B,C_stage,1,1], and ADDED to
       | that stage's feature map (broadcast over H,W) -- right after
       | that stage's conv, right before its ReLU.
       v
    added into: enc1, enc2, enc3, bottleneck, dec3, dec2, dec1
    (7 injection points; NOT out_conv, which stays a plain 1x1 projection
    to image channels)

See `sinusoidal_timestep_embedding`'s own docstring for the embedding
equation itself (symbol-by-symbol, exact reproduction of Ho et al.'s
2020 official DDPM code).

Local usage (lightweight smoke test only, no dataset download):
    python code/unet_time_conditioned_64.py --smoke-test
"""

import argparse
import math

import torch
import torch.nn as nn

from unet_tiny_denoising_64 import count_trainable_parameters


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Sinusoidal timestep embedding -- exact reproduction of the
    `get_timestep_embedding` function in Ho et al.'s (2020) official
    DDPM code (hojonathanho/diffusion, nn.py), itself borrowed from the
    Transformer positional encoding (Vaswani et al., 2017, "Attention
    Is All You Need", Sec. 3.5).

    Equation, for frequency index i = 0 .. embedding_dim/2 - 1:
        freq_i     = max_period ** ( -i / (embedding_dim/2 - 1) )
        angle_i(t) = t * freq_i
        emb(t)     = [ sin(angle_0(t)), ..., sin(angle_{d/2-1}(t)),
                       cos(angle_0(t)), ..., cos(angle_{d/2-1}(t)) ]
    i.e. embedding_dim/2 sin features concatenated with embedding_dim/2
    cos features (`[sin || cos]`), evaluated at embedding_dim/2
    geometrically spaced frequencies from 1 (i=0, fastest-oscillating)
    down to 1/max_period (i=d/2-1, slowest). Pairing sin with cos at
    each frequency is what makes the embedding unambiguous: sin(angle)
    alone is many-to-one (sin(30 deg) == sin(150 deg)), but the PAIR
    (cos(angle), sin(angle)) is a point on the unit circle, which maps
    one-to-one back to angle (mod a full turn).

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


class TimeConditionedUNet64(nn.Module):
    """3-level encoder/decoder U-Net, always conditioned on a diffusion
    timestep t. See module docstring for the full architecture and the
    "How the timestep enters the network" shape trace.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        time_embedding_dim: int = 128,
        time_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        # time_hidden_dim defaults to 4*base_channels (= c4, the
        # bottleneck width), a common DDPM convention tying the time
        # embedding's width to the network's own channel scale.
        self.time_hidden_dim = time_hidden_dim if time_hidden_dim is not None else base_channels * 4

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # --- timestep embedding path ---
        # raw sin/cos fingerprint [B,time_embedding_dim] -> learned
        # embedding [B,time_hidden_dim]; SiLU (as in Ho et al.'s code)
        # lets the network reshape the fixed sin/cos basis nonlinearly.
        self.time_mlp_fc1 = nn.Linear(time_embedding_dim, self.time_hidden_dim)
        self.time_mlp_act = nn.SiLU()
        self.time_mlp_fc2 = nn.Linear(self.time_hidden_dim, self.time_hidden_dim)

        # One projection per injection stage: time_hidden_dim -> that
        # stage's channel count, so temb can be added as a per-channel
        # bias to that stage's feature map (see `forward`'s inject_time).
        # Each width below is matched to the conv stage it feeds, defined
        # further down -- e.g. time_proj_enc1 must output c1 channels
        # because enc1's conv (below) outputs c1 channels.
        self.time_proj_enc1 = nn.Linear(self.time_hidden_dim, c1)        # matches enc1 output width (c1)
        self.time_proj_enc2 = nn.Linear(self.time_hidden_dim, c2)        # matches enc2 output width (c2)
        self.time_proj_enc3 = nn.Linear(self.time_hidden_dim, c3)        # matches enc3 output width (c3)
        self.time_proj_bottleneck = nn.Linear(self.time_hidden_dim, c4)  # matches bottleneck output width (c4)
        self.time_proj_dec3 = nn.Linear(self.time_hidden_dim, c3)        # matches dec3 output width (c3)
        self.time_proj_dec2 = nn.Linear(self.time_hidden_dim, c2)        # matches dec2 output width (c2)
        self.time_proj_dec1 = nn.Linear(self.time_hidden_dim, c1)        # matches dec1 output width (c1)

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

    def forward(self, x: torch.Tensor, t: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """x: [B, in_channels, 64, 64] (the noisy image, x_noisy).
        t: [B] diffusion timesteps, one integer per batch element --
        always required (unlike TinyUNet64, there is no "off" switch
        here; that is the point of this file).
        Returns eps_hat: [B, out_channels, 64, 64], the predicted noise
        (unbounded, no final activation)."""

        def log(label, tensor):
            if verbose:
                print(f"{label:<48}: {tuple(tensor.shape)}")

        assert t.shape == (x.shape[0],), f"t must be [B]=({x.shape[0]},), got {tuple(t.shape)}"
        log("input x", x)
        log("input t (raw, one int per batch element)", t)

        # --- build the time embedding ONCE, reuse it at every stage below ---
        # Step 1: fixed sin/cos "fingerprint" of t -- deterministic math,
        # zero learned parameters, see sinusoidal_timestep_embedding's docstring.
        raw_temb = sinusoidal_timestep_embedding(t, self.time_embedding_dim)  # [B] -> [B,time_embedding_dim]
        log("[TIME ENTERS HERE] sinusoidal embedding of t", raw_temb)

        # Step 2: let the network LEARN what to do with that fingerprint.
        # This 2-layer MLP is the only trainable part of the embedding
        # path itself (the per-stage time_proj_* layers below are separate).
        temb = self.time_mlp_fc1(raw_temb)   # [B,time_embedding_dim] -> [B,time_hidden_dim]
        temb = self.time_mlp_act(temb)
        temb = self.time_mlp_fc2(temb)       # [B,time_hidden_dim] -> [B,time_hidden_dim]
        log("time embedding after time_mlp (fc1->SiLU->fc2)", temb)

        def inject_time(h: torch.Tensor, proj: nn.Linear, label: str) -> torch.Tensor:
            """h: [B,C,H,W] -> h + proj(temb) broadcast over H,W. This
            is the point where timestep information enters the
            convolutional feature map: proj(temb) gives a per-sample,
            per-channel bias [B,C]; [:, :, None, None] reshapes it to
            [B,C,1,1] so PyTorch's broadcasting rule adds the SAME bias
            to every one of that channel's H*W spatial positions (t
            carries no spatial information of its own, so none is
            introduced here)."""
            time_bias = proj(temb)                   # [B,time_hidden_dim] -> [B,C]
            log(f"  [TIME ENTERS HERE] {label}: time_proj(temb)", time_bias)
            time_bias = time_bias[:, :, None, None]  # [B,C] -> [B,C,1,1]
            h = h + time_bias                        # [B,C,H,W] + [B,C,1,1] -> [B,C,H,W]
            log(f"  {label}: feature map after + time embedding", h)
            return h

        # --- encoder ---
        # Every stage follows the SAME 3-step pattern: conv -> inject
        # time -> activation. The conv first extracts spatial features;
        # the timestep bias is added while the map is still in its raw,
        # unbounded conv-output range (adding it after a ReLU would mean
        # a negative time bias could never "undo" a clamped-to-zero
        # pixel); only then does the nonlinearity get applied.
        h = self.enc1(x)                                  # conv
        h = inject_time(h, self.time_proj_enc1, "enc1")   # + time
        skip1 = self.relu_e1(h)                           # activation; also the tensor kept for the U-Net skip
        log("skip1 saved (before pool1)", skip1)
        h = self.pool1(skip1)
        log("after pool1", h)

        h = self.enc2(h)
        h = inject_time(h, self.time_proj_enc2, "enc2")
        skip2 = self.relu_e2(h)
        log("skip2 saved (before pool2)", skip2)
        h = self.pool2(skip2)
        log("after pool2", h)

        h = self.enc3(h)
        h = inject_time(h, self.time_proj_enc3, "enc3")
        skip3 = self.relu_e3(h)
        log("skip3 saved (before pool3)", skip3)
        h = self.pool3(skip3)
        log("after pool3", h)

        # --- bottleneck: lowest resolution (8x8), widest channels (c4) ---
        h = self.bottleneck(h)
        h = inject_time(h, self.time_proj_bottleneck, "bottleneck")
        h = self.relu_b(h)
        log("after bottleneck", h)

        # --- decoder, stage 3: upsample -> concat skip3 -> conv -> +time -> relu ---
        # Concatenation happens BEFORE the time injection: dec3's conv
        # first fuses the upsampled decoder features with the encoder's
        # skip3 features (channel-wise), and only the result of that
        # fusion gets the timestep bias -- t does not touch the skip
        # connection itself.
        h = self.up3(h)
        log("up3 output (before concat with skip3)", h)
        log("skip3 (being concatenated)", skip3)
        h = torch.cat([h, skip3], dim=1)  # [B,c3,16,16]+[B,c3,16,16] -> [B,2*c3,16,16] along channel dim
        log("after concat(up3, skip3)", h)
        h = self.dec3(h)
        h = inject_time(h, self.time_proj_dec3, "dec3")
        h = self.relu_d3(h)
        log("after dec3", h)

        # --- decoder, stage 2: upsample -> concat skip2 -> conv -> +time -> relu ---
        h = self.up2(h)
        log("up2 output (before concat with skip2)", h)
        log("skip2 (being concatenated)", skip2)
        h = torch.cat([h, skip2], dim=1)  # [B,c2,32,32]+[B,c2,32,32] -> [B,2*c2,32,32]
        log("after concat(up2, skip2)", h)
        h = self.dec2(h)
        h = inject_time(h, self.time_proj_dec2, "dec2")
        h = self.relu_d2(h)
        log("after dec2", h)

        # --- decoder, stage 1: upsample -> concat skip1 -> conv -> +time -> relu ---
        h = self.up1(h)
        log("up1 output (before concat with skip1)", h)
        log("skip1 (being concatenated)", skip1)
        h = torch.cat([h, skip1], dim=1)  # [B,c1,64,64]+[B,c1,64,64] -> [B,2*c1,64,64]
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

    model = TimeConditionedUNet64(in_channels=3, out_channels=3, base_channels=32).to(device)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    dummy_noisy_images = torch.rand(4, 3, 64, 64, device=device)  # [B,3,64,64], in [0,1]
    dummy_true_noise = torch.randn(4, 3, 64, 64, device=device)   # [B,3,64,64], the eps the model should predict
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
        "This file only demonstrates the time-conditioned U-Net architecture "
        "(use --smoke-test). It is not wired to a training loop or a noise "
        "schedule yet -- see code/beta_schedule.py and code/ddpm_forward_process.py "
        "for the pieces that would need to be connected to actually train this model."
    )


if __name__ == "__main__":
    main()
