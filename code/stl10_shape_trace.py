"""Standalone shape-trace utility for the two STL-10 reconstruction
models compared in code/compare_stl10_autoencoder_vs_unet.py.

Kept in its own small file (rather than only inside the larger
comparison script) specifically so it can be imported independently --
useful if a Colab session ends up with a stale/cached copy of the
bigger file but re-clones this one fresh.

Local usage (lightweight, no dataset download -- just prints shapes):
    python code/stl10_shape_trace.py
"""

import torch

from stl10_conv_autoencoder import STL10ConvAutoencoder
from stl10_unet import SmallUNet


def print_shape_traces(device: torch.device, latent_dim: int = 128, batch_size: int = 4):
    """One verbose forward pass per model on a synthetic batch, purely
    to print the full [B,C,H,W] shape trace -- including, for the
    U-Net, every skip-connection save and every concatenation -- before
    training starts. Builds fresh, untrained, throwaway instances of
    both models just for this printout: any later training models
    (e.g. inside compare_stl10_autoencoder_vs_unet.run_comparison) are
    separate instances with their own seeded construction, so this has
    no effect on reproducibility of an actual training run."""
    sample_batch = torch.rand(batch_size, 3, 96, 96, device=device)  # [B,3,96,96], synthetic, in [0,1]

    print("=== STL10ConvAutoencoder: full shape trace ===")
    autoencoder_probe = STL10ConvAutoencoder(latent_dim=latent_dim).to(device)
    _ = autoencoder_probe(sample_batch, verbose=True)

    print("\n=== SmallUNet: full shape trace (skip connections + concatenations) ===")
    unet_probe = SmallUNet().to(device)
    _ = unet_probe(sample_batch, verbose=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print_shape_traces(device)
