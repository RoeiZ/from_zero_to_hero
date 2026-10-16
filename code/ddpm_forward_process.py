"""DDPM closed-form forward (noising) process only -- no neural network.

Paper: Ho et al., 2020, "Denoising Diffusion Probabilistic Models".
The forward process is a fixed Markov chain q(x_t | x_{t-1}) (Eq. 2).
Iterating it from x_0 gives a closed-form "jump straight to step t"
formula (paper Eq. 4, reparameterized with the Gaussian sampling
trick):

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)

i.e. q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I).

Symbol -> code variable -> shape -> meaning:
    x_0                          x0                          [C,H,W]   clean image, DDPM convention: pixels in [-1,1]
    t (0-indexed array index)    t_indices                  [K]       which timesteps to sample x_t at (K <= T)
    alpha_bar_t                  alpha_bars[t_indices]       [K]       cumulative signal-retention factor (code/beta_schedule.py)
    sqrt(alpha_bar_t)            sqrt_alpha_bar_t            [K]       "signal coefficient": weight on x_0
    sqrt(1-alpha_bar_t)          sqrt_one_minus_alpha_bar_t  [K]       "noise coefficient": weight on eps
    eps                          noise                       [K,C,H,W] standard Gaussian noise, same shape as x_0 (broadcast over K)
    x_t                          x_t                         [K,C,H,W] noised image at each requested t

Indexing convention: identical to code/beta_schedule.py -- t_indices
are plain 0-indexed positions into the length-T alpha_bars array
(index 0 <-> paper's t=1, index T-1 <-> paper's t=T). A single shared
x_0 is broadcast against K different timesteps requested at once (one
eps draw per requested t, all independent).

This file implements ONLY the forward process above. No model, no
training, no loss -- just the closed-form noising equation and its two
scalar coefficients, applied to one synthetic image.

Local usage (lightweight, CPU, no dataset, no GPU):
    python code/ddpm_forward_process.py --smoke-test
"""

import argparse

import torch
import matplotlib.pyplot as plt

from beta_schedule import linear_beta_schedule, compute_alpha_and_alpha_bar


def make_synthetic_clean_image(height: int = 64, width: int = 64) -> torch.Tensor:
    """Procedurally generated image, no dataset download: a smooth
    diagonal color gradient background with a solid-color circle in
    the center (sharp edges -- gives the noising process real
    structure to destroy, instead of a flat image).

    Returns:
        x0: [3,H,W] float32, pixel values in [-1,1] (DDPM convention).
    """
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )  # y, x: [H,W] each, coordinate grids in [-1,1]

    # Smooth background: one gradient direction per RGB channel, in [0,1].
    red = 0.5 + 0.5 * x
    green = 0.5 + 0.5 * y
    blue = 0.5 - 0.5 * x * y
    background = torch.stack([red, green, blue], dim=0)  # [3,H,W], in [0,1]

    # Sharp circle (solid color) in the center, high-frequency edge content.
    radius = 0.4
    distance_from_center = torch.sqrt(x**2 + y**2)  # [H,W]
    circle_mask = (distance_from_center < radius).float().unsqueeze(0)  # [1,H,W], broadcasts over 3 channels
    circle_color = torch.tensor([0.9, 0.1, 0.1]).view(3, 1, 1)  # [3,1,1]

    image_01 = background * (1 - circle_mask) + circle_color * circle_mask  # [3,H,W], in [0,1]
    x0 = image_01 * 2.0 - 1.0  # rescale [0,1] -> [-1,1] (DDPM convention)
    return x0.clamp(-1.0, 1.0)


def get_signal_and_noise_coefficients(
    t_indices: torch.Tensor, alpha_bars: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """sqrt(alpha_bar_t) (signal coefficient) and sqrt(1-alpha_bar_t)
    (noise coefficient) for each requested t.

    Args:
        t_indices: [K] long tensor, 0-indexed into alpha_bars
        alpha_bars: [T]

    Returns:
        sqrt_alpha_bar_t:           [K]
        sqrt_one_minus_alpha_bar_t: [K]
    """
    alpha_bar_t = alpha_bars[t_indices]  # [K], gather: one alpha_bar value per requested t
    sqrt_alpha_bar_t = alpha_bar_t.sqrt()
    sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt()
    return sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t


def forward_diffusion_sample(
    x0: torch.Tensor,
    t_indices: torch.Tensor,
    alpha_bars: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot closed-form noising: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps.

    Args:
        x0:         [C,H,W], single clean image, pixels in [-1,1]
        t_indices:  [K], 0-indexed timesteps to sample (shared x0, K independent eps draws)
        alpha_bars: [T]
        generator:  optional torch.Generator, for reproducible noise

    Returns:
        x_t:                        [K,C,H,W]
        noise:                      [K,C,H,W], the eps actually drawn
        sqrt_alpha_bar_t:           [K]
        sqrt_one_minus_alpha_bar_t: [K]
    """
    num_requested_t = t_indices.shape[0]
    num_channels, height, width = x0.shape

    sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = get_signal_and_noise_coefficients(t_indices, alpha_bars)  # [K], [K]

    noise = torch.randn(
        num_requested_t, num_channels, height, width, generator=generator
    )  # [K,C,H,W], eps ~ N(0,I), independent draw per requested t

    x0_broadcast = x0.unsqueeze(0)  # [1,C,H,W] -> broadcasts over the K requested timesteps
    signal_coefficient = sqrt_alpha_bar_t.view(num_requested_t, 1, 1, 1)  # [K,1,1,1] -> broadcasts over C,H,W
    noise_coefficient = sqrt_one_minus_alpha_bar_t.view(num_requested_t, 1, 1, 1)  # [K,1,1,1]

    x_t = signal_coefficient * x0_broadcast + noise_coefficient * noise  # [K,C,H,W]
    return x_t, noise, sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t


def signal_noise_table(
    t_indices: torch.Tensor,
    sqrt_alpha_bar_t: torch.Tensor,
    sqrt_one_minus_alpha_bar_t: torch.Tensor,
) -> str:
    """Formats (t, signal coefficient, noise coefficient) for the
    requested timesteps as a printable table."""
    lines = [f"{'t':>6} {'sqrt(alpha_bar_t)':>20} {'sqrt(1-alpha_bar_t)':>22}"]
    lines[0] += "   (signal coeff.)      (noise coeff.)"
    for i, t in enumerate(t_indices.tolist()):
        lines.append(
            f"{t:>6} {sqrt_alpha_bar_t[i].item():>20.6f} {sqrt_one_minus_alpha_bar_t[i].item():>22.6f}"
        )
    return "\n".join(lines)


def visualize_forward_process(
    x0: torch.Tensor,
    x_t: torch.Tensor,
    t_indices: torch.Tensor,
    sqrt_alpha_bar_t: torch.Tensor,
    sqrt_one_minus_alpha_bar_t: torch.Tensor,
    save_path: str | None = None,
):
    """One row: x0 followed by x_t at each requested timestep, titled
    with t and its signal/noise coefficients. Display only: pixels are
    mapped from [-1,1] back to [0,1] and clamped, since x_t can fall
    outside [-1,1] once noise is added (clamping is for `imshow` only,
    not applied to the underlying math)."""
    num_requested_t = t_indices.shape[0]
    fig, axes = plt.subplots(1, num_requested_t + 1, figsize=(2.3 * (num_requested_t + 1), 2.9))

    def to_displayable(image_chw: torch.Tensor):
        return ((image_chw.clamp(-1.0, 1.0) + 1.0) / 2.0).permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C], in [0,1]

    axes[0].imshow(to_displayable(x0))
    axes[0].set_title("x0\n(signal=1, noise=0)", fontsize=8)
    axes[0].axis("off")

    for i, t in enumerate(t_indices.tolist()):
        axes[i + 1].imshow(to_displayable(x_t[i]))
        axes[i + 1].set_title(
            f"t={t}\nsignal={sqrt_alpha_bar_t[i].item():.3f}\nnoise={sqrt_one_minus_alpha_bar_t[i].item():.3f}",
            fontsize=8,
        )
        axes[i + 1].axis("off")

    fig.suptitle("Forward process: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps", y=1.08)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved forward-process figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _smoke_test():
    """Lightweight local checks: shape, dtype, finiteness, the exact
    identity signal_coeff^2 + noise_coeff^2 == 1, the two limiting
    cases (t near 0 -> x_t approx x0; t near T-1 -> x_t approx pure
    noise), and fixed-seed reproducibility. No dataset, no GPU."""
    num_timesteps = 1000
    betas = linear_beta_schedule(num_timesteps)
    _alphas, alpha_bars = compute_alpha_and_alpha_bar(betas)

    x0 = make_synthetic_clean_image(32, 32)
    assert x0.shape == (3, 32, 32)
    assert x0.min() >= -1.0 and x0.max() <= 1.0

    t_indices = torch.tensor([0, 100, 300, 400, 600, 800, 999])
    generator = torch.Generator().manual_seed(0)
    x_t, noise, sqrt_ab, sqrt_omab = forward_diffusion_sample(x0, t_indices, alpha_bars, generator=generator)

    assert x_t.shape == (7, 3, 32, 32)
    assert noise.shape == (7, 3, 32, 32)
    assert torch.isfinite(x_t).all()

    # Exact identity: Var(x_t) = alpha_bar_t * Var(x0) + (1-alpha_bar_t) * Var(eps)
    # only holds in distribution: alpha_bar_t and (1-alpha_bar_t) themselves sum to 1.
    assert torch.allclose(sqrt_ab**2 + sqrt_omab**2, torch.ones_like(sqrt_ab), atol=1e-6)

    # Limiting case t=0 (0-indexed first step, alpha_bar_0 close to 1): x_t approx x0.
    assert sqrt_ab[0].item() > 0.99 and sqrt_omab[0].item() < 0.02
    reconstruction_error_t0 = (x_t[0] - x0).abs().max().item()
    assert reconstruction_error_t0 < 0.05, "t=0 should barely perturb x0"

    # Limiting case t=999=T-1 (alpha_bar close to 0): x_t approx sqrt(1-alpha_bar_t)*eps alone.
    assert sqrt_ab[-1].item() < 0.02
    pure_noise_component = sqrt_omab[-1] * noise[-1]
    assert torch.allclose(x_t[-1], sqrt_ab[-1] * x0 + pure_noise_component, atol=1e-6)

    # Fixed-seed reproducibility: same generator seed -> identical noise -> identical x_t.
    generator_repeat = torch.Generator().manual_seed(0)
    x_t_repeat, _noise_repeat, _sab, _somab = forward_diffusion_sample(x0, t_indices, alpha_bars, generator=generator_repeat)
    assert torch.equal(x_t, x_t_repeat)

    print("ddpm_forward_process smoke test passed.")
    print(signal_noise_table(t_indices, sqrt_ab, sqrt_omab))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run lightweight local checks and exit")
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        betas = linear_beta_schedule(args.num_timesteps)
        _alphas, alpha_bars = compute_alpha_and_alpha_bar(betas)

        x0 = make_synthetic_clean_image(64, 64)
        t_indices = torch.tensor([0, 100, 300, 400, 600, 800, 999])
        generator = torch.Generator().manual_seed(args.seed)
        x_t, _noise, sqrt_ab, sqrt_omab = forward_diffusion_sample(x0, t_indices, alpha_bars, generator=generator)

        print(signal_noise_table(t_indices, sqrt_ab, sqrt_omab))
        visualize_forward_process(x0, x_t, t_indices, sqrt_ab, sqrt_omab)
