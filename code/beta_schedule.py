"""DDPM forward-process linear beta schedule (Ho et al., 2020, "Denoising
Diffusion Probabilistic Models", Sec. 2 and Sec. 4 "Experimental details").

Forward process (paper Eq. 2): for t = 1..T,
    q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
beta_t in (0,1) is a fixed (not learned) variance schedule. The paper
sets it to increase linearly from beta_1 = 1e-4 to beta_T = 0.02.

Two quantities derived from beta_t are used throughout DDPM (paper
Eq. 4 and surrounding text):
    alpha_t      := 1 - beta_t                  (per-step signal-retention factor)
    alpha_bar_t  := prod_{s=1}^{t} alpha_s        (cumulative signal-retention factor)
alpha_bar_t is what gives the closed-form one-shot noising formula
    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
(not implemented here -- this module only builds and visualizes the
schedule itself; the one-shot noising formula is a later exercise).

Indexing convention used in this file: all returned tensors are
0-indexed arrays of length T (Python/torch convention), where array
index i = 0 corresponds to the paper's t = 1, and index i = T-1
corresponds to the paper's t = T. `schedule_table` below prints the
paper's 1-indexed t for readability.

Local usage (lightweight, CPU, no dataset, no GPU):
    python code/beta_schedule.py --smoke-test
"""

import argparse

import torch
import matplotlib.pyplot as plt


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """beta_t, linearly spaced from beta_start (=beta_1) to beta_end (=beta_T).

    Returns:
        betas: [T] float32 tensor, betas[0] == beta_start, betas[-1] == beta_end.
    """
    return torch.linspace(beta_start, beta_end, num_timesteps, device=device, dtype=torch.float32)


def compute_alpha_and_alpha_bar(betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """alpha_t = 1 - beta_t, and alpha_bar_t = cumprod_{s<=t} alpha_s.

    Args:
        betas: [T]

    Returns:
        alphas:     [T], alphas[i]     = 1 - betas[i]
        alpha_bars: [T], alpha_bars[i] = prod_{j=0}^{i} alphas[j]
    """
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars


def schedule_table(
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    num_rows: int = 10,
) -> str:
    """Formats `num_rows` evenly spaced (t, beta_t, alpha_t, alpha_bar_t)
    rows (always including the first and last timestep) as a printable
    table, with t shown in the paper's 1-indexed convention."""
    num_timesteps = betas.shape[0]
    num_rows = min(num_rows, num_timesteps)
    row_indices = torch.linspace(0, num_timesteps - 1, num_rows).round().long().unique()

    lines = [f"{'t':>6} {'beta_t':>12} {'alpha_t':>12} {'alpha_bar_t':>14}"]
    for i in row_indices.tolist():
        t_1indexed = i + 1
        lines.append(
            f"{t_1indexed:>6} {betas[i].item():>12.6f} {alphas[i].item():>12.6f} {alpha_bars[i].item():>14.6e}"
        )
    return "\n".join(lines)


def plot_beta_alpha_alphabar(
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    save_path: str | None = None,
):
    """Three-panel figure, one per schedule quantity, all against the
    1-indexed timestep t: beta_t (linearly increasing), alpha_t = 1 -
    beta_t (linearly decreasing, mirrors beta_t), and alpha_bar_t
    (cumulative product -- decays roughly exponentially, so also shown
    on a log-y-scale inset-free second line via a twin panel)."""
    num_timesteps = betas.shape[0]
    t_axis = torch.arange(1, num_timesteps + 1)  # 1-indexed t, matches paper

    fig, (ax_beta, ax_alpha, ax_alphabar) = plt.subplots(1, 3, figsize=(14, 4.2))

    ax_beta.plot(t_axis, betas, color="tab:blue")
    ax_beta.set_xlabel("t")
    ax_beta.set_ylabel("beta_t")
    ax_beta.set_title("beta_t\n(linear schedule, chosen variance)")

    ax_alpha.plot(t_axis, alphas, color="tab:orange")
    ax_alpha.set_xlabel("t")
    ax_alpha.set_ylabel("alpha_t = 1 - beta_t")
    ax_alpha.set_title("alpha_t\n(per-step signal retained)")

    ax_alphabar.plot(t_axis, alpha_bars, color="tab:green")
    ax_alphabar.set_xlabel("t")
    ax_alphabar.set_ylabel("alpha_bar_t")
    ax_alphabar.set_title("alpha_bar_t = cumprod(alpha)\n(cumulative signal retained)")
    ax_alphabar.set_yscale("log")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved beta/alpha/alpha_bar figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _smoke_test():
    """Lightweight local checks: shape, dtype, finiteness, monotonicity,
    range, and the two limiting-case identities alpha_t = 1 - beta_t
    and alpha_bar_t = prod_{s<=t} alpha_s. No dataset, no GPU."""
    num_timesteps = 1000
    betas = linear_beta_schedule(num_timesteps)
    alphas, alpha_bars = compute_alpha_and_alpha_bar(betas)

    assert betas.shape == (num_timesteps,), betas.shape
    assert alphas.shape == (num_timesteps,), alphas.shape
    assert alpha_bars.shape == (num_timesteps,), alpha_bars.shape

    assert torch.isfinite(betas).all()
    assert torch.isfinite(alphas).all()
    assert torch.isfinite(alpha_bars).all()

    assert torch.allclose(betas[0], torch.tensor(1e-4))
    assert torch.allclose(betas[-1], torch.tensor(0.02))
    assert (betas[1:] >= betas[:-1]).all(), "beta_t must be non-decreasing (linear schedule)"

    assert torch.allclose(alphas, 1.0 - betas), "alpha_t must equal 1 - beta_t exactly"
    assert (alphas > 0).all() and (alphas < 1).all()

    # alpha_bar_t is a cumulative product of values in (0,1) -> strictly decreasing.
    assert (alpha_bars[1:] < alpha_bars[:-1]).all(), "alpha_bar_t must be strictly decreasing"
    assert torch.allclose(alpha_bars[0], alphas[0]), "alpha_bar_1 must equal alpha_1"
    # Independent re-derivation via torch.prod, not cumprod, as a cross-check at t=T.
    assert torch.allclose(alpha_bars[-1], torch.prod(alphas), atol=1e-6, rtol=1e-4)
    assert (alpha_bars > 0).all() and (alpha_bars <= 1).all()

    print("beta_schedule smoke test passed.")
    print(schedule_table(betas, alphas, alpha_bars, num_rows=10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run lightweight local checks and exit")
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        betas = linear_beta_schedule(args.num_timesteps, args.beta_start, args.beta_end)
        alphas, alpha_bars = compute_alpha_and_alpha_bar(betas)
        print(schedule_table(betas, alphas, alpha_bars, num_rows=10))
        plot_beta_alpha_alphabar(betas, alphas, alpha_bars)
