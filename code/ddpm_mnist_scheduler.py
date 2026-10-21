"""DDPM noise scheduler (Ho et al., 2020, "Denoising Diffusion
Probabilistic Models") -- the piece that owns the fixed forward-process
math (beta_t, alpha_t, alpha_bar_t) and both directions built on top of
it: closed-form forward noising (used during training) and the
single-step reverse update (used during sampling). No neural network
lives in this file; `code/ddpm_mnist_model.py` is the network that
predicts eps, this file is purely the fixed (non-learned) schedule
around it.

Reuses code/beta_schedule.py for beta_t/alpha_t/alpha_bar_t instead of
recomputing them -- see that file for the schedule derivation itself.

--- 1. Forward process (training target), paper Eq. 4 reparameterized ---

    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0,I)

This is the same equation as code/ddpm_forward_process.py, but with a
DIFFERENT indexing convention: that file samples ONE image at K
different timesteps (shape [K,C,H,W] from a single [C,H,W] image).
Training instead samples a whole BATCH of images, each at its OWN
independently drawn random timestep (shape [B,C,H,W] with t: [B], one
timestep per batch element) -- see `DDPMScheduler.add_noise`.

--- 2. Reverse process (sampling), paper Algorithm 2 ---

For t = T, ..., 1 (this file's 0-indexed t_index = T-1, ..., 0):

    x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_hat ) + sigma_t * z,
    z ~ N(0,I) if t_index > 0 else z = 0

eps_hat = model(x_t, t) is the network's noise prediction. sigma_t^2 is
the fixed (non-learned) reverse-process variance; this file uses the
paper's Eq. 7 posterior variance

    sigma_t^2 = beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t,

i.e. the variance of the true posterior q(x_{t-1} | x_t, x0), one of
the two fixed-variance choices Ho et al. compare (the other being
sigma_t^2 = beta_t itself; the paper reports both work similarly --
this file picks beta_tilde_t since it is the mathematically exact
posterior variance, not an upper bound on it).

Symbol -> code variable -> shape:
    beta_t                  betas[i]              scalar (gathered at an index i)
    alpha_t                 alphas[i]             scalar
    alpha_bar_t             alpha_bars[i]         scalar
    alpha_bar_{t-1}         alpha_bars_prev[i]    scalar, := 1.0 when i == 0 (paper's alpha_bar_0 := 1)
    beta_tilde_t            posterior_variance[i] scalar

See `make_respaced_schedule` for sampling with FEWER reverse steps than
training used (the "respacing" trick, Nichol & Dhariwal, 2021,
"Improved Denoising Diffusion Probabilistic Models", Sec. 4.1): a
well-defined sub-schedule, not a naive "just skip steps".

Local usage (lightweight smoke test only, no dataset, no GPU):
    python code/ddpm_mnist_scheduler.py --smoke-test
"""

import argparse
from dataclasses import dataclass

import torch

from beta_schedule import linear_beta_schedule, compute_alpha_and_alpha_bar


def _alpha_bars_prev_and_posterior_variance(
    alpha_bars: torch.Tensor, betas: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """alpha_bar_{t-1} (with alpha_bar_{-1} := 1, paper's alpha_bar_0 := 1)
    and the Eq. 7 posterior variance beta_tilde_t, both length [T] (or
    [S] for a respaced sub-schedule -- this helper is shape-agnostic).
    """
    alpha_bars_prev = torch.cat([torch.ones(1, device=alpha_bars.device, dtype=alpha_bars.dtype), alpha_bars[:-1]])
    posterior_variance = (1.0 - alpha_bars_prev) / (1.0 - alpha_bars) * betas
    # At i=0, alpha_bars_prev=1 and alpha_bars[0]=alpha_bar_t at the first
    # kept timestep; if that first timestep IS index 0 then alpha_bars[0]
    # also equals 1 - beta[0]*small terms, never exactly 1, so no 0/0 --
    # the only exact-1 case is alpha_bars_prev itself, which is an input,
    # not a divisor.
    return alpha_bars_prev, posterior_variance


def ddpm_reverse_step(
    x_t: torch.Tensor,
    eps_hat: torch.Tensor,
    beta_t: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    posterior_variance_t: torch.Tensor,
    add_noise: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One reverse-process update, paper Algorithm 2's single line,
    shared by both the full schedule and any respaced sub-schedule
    (both just gather different scalars to pass in here).

    Args:
        x_t:    [B,C,H,W], current noisy sample
        eps_hat: [B,C,H,W], model's predicted noise at this t
        beta_t, alpha_t, alpha_bar_t, posterior_variance_t: scalars (0-d tensors)
        add_noise: False at the final step (t_index==0 / i==0), matching
            the paper's "z=0 when t=1"; True otherwise.
        generator: optional torch.Generator for reproducible sampling

    Returns:
        x_prev: [B,C,H,W], i.e. x_{t-1}
    """
    noise_coefficient = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)  # scalar
    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - noise_coefficient * eps_hat)  # [B,C,H,W]

    if add_noise:
        z = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)  # [B,C,H,W]
        x_prev = mean + torch.sqrt(posterior_variance_t) * z
    else:
        x_prev = mean
    return x_prev


class DDPMScheduler:
    """Owns the full-length (T-step) linear beta schedule and the two
    operations built on it: `add_noise` (forward, training) and
    `reverse_step` (one reverse update, sampling). See module docstring
    for the equations.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str | None = None,
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = linear_beta_schedule(num_timesteps, beta_start, beta_end, device=device)  # [T]
        self.alphas, self.alpha_bars = compute_alpha_and_alpha_bar(self.betas)                  # [T], [T]
        self.alpha_bars_prev, self.posterior_variance = _alpha_bars_prev_and_posterior_variance(
            self.alpha_bars, self.betas
        )  # [T], [T]

    def to(self, device: torch.device | str) -> "DDPMScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.alpha_bars_prev = self.alpha_bars_prev.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def config(self) -> dict:
        """Plain-dict config for checkpointing (CLAUDE.md: checkpoints
        must record the scheduler state needed to reconstruct it)."""
        return {
            "num_timesteps": self.num_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }

    def add_noise(
        self,
        x0: torch.Tensor,
        t_indices: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form forward noising, ONE independently-sampled t per
        batch element (training use case -- contrast with
        code/ddpm_forward_process.py's one-image/many-t convention,
        see module docstring).

        Args:
            x0:        [B,C,H,W], clean images, pixels in [-1,1]
            t_indices: [B], 0-indexed timestep per batch element
            noise:     optional pre-drawn [B,C,H,W] eps; drawn fresh if None

        Returns:
            x_t:   [B,C,H,W]
            noise: [B,C,H,W], the eps actually used (== training target)
        """
        batch_size = x0.shape[0]
        assert t_indices.shape == (batch_size,), f"t_indices must be [B]=({batch_size},), got {tuple(t_indices.shape)}"

        alpha_bar_t = self.alpha_bars[t_indices]  # [B], gather: one alpha_bar value per batch element
        sqrt_alpha_bar_t = alpha_bar_t.sqrt().view(batch_size, 1, 1, 1)             # [B,1,1,1] -> broadcasts over C,H,W
        sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt().view(batch_size, 1, 1, 1)  # [B,1,1,1]

        if noise is None:
            noise = torch.randn_like(x0)  # [B,C,H,W]

        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise  # [B,C,H,W]
        return x_t, noise

    def reverse_step(
        self,
        x_t: torch.Tensor,
        t_index: int,
        eps_hat: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One reverse update at 0-indexed timestep `t_index` (shared
        across the whole batch -- standard ancestral DDPM sampling uses
        the same t for every sample at a given step).

        Args:
            x_t:      [B,C,H,W]
            t_index:  int in [0, num_timesteps-1]
            eps_hat:  [B,C,H,W], model(x_t, t) output
            generator: optional torch.Generator

        Returns:
            x_prev: [B,C,H,W]
        """
        return ddpm_reverse_step(
            x_t,
            eps_hat,
            beta_t=self.betas[t_index],
            alpha_t=self.alphas[t_index],
            alpha_bar_t=self.alpha_bars[t_index],
            posterior_variance_t=self.posterior_variance[t_index],
            add_noise=(t_index > 0),
            generator=generator,
        )

    def make_respaced_schedule(self, num_inference_steps: int) -> "RespacedSchedule":
        """Build a sub-schedule using only `num_inference_steps` of the
        original T timesteps, for cheaper sampling (fewer network
        evaluations) from the SAME trained model.

        This is the standard "respacing" reparameterization (Nichol &
        Dhariwal, 2021, Sec. 4.1; also used in OpenAI's guided-diffusion
        `space_timesteps`): pick an increasing subsequence of original
        timesteps tau_0 < tau_1 < ... < tau_{S-1} (always including the
        first and last original timestep), and define NEW betas for the
        shorter chain from the ratio of consecutive alpha_bar values at
        those indices:

            alpha_bar'_i := alpha_bar_{tau_i}                      (unchanged: same marginal as the original chain)
            beta'_i      := 1 - alpha_bar'_i / alpha_bar'_{i-1}    (alpha_bar'_{-1} := 1)

        This keeps the marginal q(x_{tau_i} | x0) IDENTICAL to the
        original T-step chain at every kept timestep -- only the
        Markov chain CONNECTING those marginals becomes coarser (fewer,
        larger reverse jumps), which is exactly the extra discretization
        error this exercise's step-count comparison is measuring.

        Args:
            num_inference_steps: S, how many reverse steps to use (S <= T)

        Returns:
            RespacedSchedule with S-length betas/alphas/alpha_bars/
            posterior_variance and the `timestep_map` [S] of original
            (0-indexed) timesteps to feed the MODEL at each step (the
            model was trained on original-scale t, so it must still be
            queried with the ORIGINAL timestep value, not the sub-schedule
            index i).
        """
        assert 1 <= num_inference_steps <= self.num_timesteps, (
            f"num_inference_steps must be in [1,{self.num_timesteps}], got {num_inference_steps}"
        )
        timestep_map = (
            torch.linspace(0, self.num_timesteps - 1, num_inference_steps)
            .round()
            .long()
            .unique()
        )  # [S] (unique() can shrink S slightly if num_inference_steps is large relative to rounding collisions)

        alpha_bars = self.alpha_bars[timestep_map]  # [S]
        alpha_bars_prev = torch.cat([torch.ones(1, device=alpha_bars.device, dtype=alpha_bars.dtype), alpha_bars[:-1]])
        betas = 1.0 - alpha_bars / alpha_bars_prev  # [S], respaced beta'_i
        alphas = 1.0 - betas                        # [S], = alpha_bars/alpha_bars_prev exactly
        posterior_variance = (1.0 - alpha_bars_prev) / (1.0 - alpha_bars) * betas  # [S]

        return RespacedSchedule(
            timestep_map=timestep_map,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            posterior_variance=posterior_variance,
        )


@dataclass
class RespacedSchedule:
    """Short (S-step) sub-schedule built by `DDPMScheduler.make_respaced_schedule`.
    Same reverse-step math as DDPMScheduler, just indexed 0..S-1 into the
    respaced tensors instead of 0..T-1 into the full ones."""

    timestep_map: torch.Tensor        # [S], original 0-indexed timesteps
    betas: torch.Tensor               # [S]
    alphas: torch.Tensor              # [S]
    alpha_bars: torch.Tensor          # [S]
    posterior_variance: torch.Tensor  # [S]

    @property
    def num_inference_steps(self) -> int:
        return self.timestep_map.shape[0]

    def reverse_step(
        self,
        x_t: torch.Tensor,
        i: int,
        eps_hat: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """i is an index into the SHORT (0..S-1) respaced arrays, not an
        original timestep -- use `self.timestep_map[i]` to get the
        original t to feed the model (see module/class docstrings)."""
        return ddpm_reverse_step(
            x_t,
            eps_hat,
            beta_t=self.betas[i],
            alpha_t=self.alphas[i],
            alpha_bar_t=self.alpha_bars[i],
            posterior_variance_t=self.posterior_variance[i],
            add_noise=(i > 0),
            generator=generator,
        )


def _smoke_test():
    """Lightweight local checks: shapes, finiteness, the signal^2+noise^2==1
    identity, one full add_noise/reverse_step roundtrip, and the
    respaced schedule's defining invariant (its alpha_bar values match
    the original schedule EXACTLY at every kept timestep). No dataset,
    no GPU."""
    torch.manual_seed(0)
    num_timesteps = 200
    scheduler = DDPMScheduler(num_timesteps=num_timesteps)

    assert scheduler.betas.shape == (num_timesteps,)
    assert scheduler.posterior_variance.shape == (num_timesteps,)
    assert torch.isfinite(scheduler.posterior_variance).all()
    assert (scheduler.posterior_variance >= 0).all(), "variance must be non-negative"

    # --- forward (training) direction ---
    batch_size = 4
    x0 = torch.rand(batch_size, 1, 28, 28) * 2 - 1  # [B,1,28,28], in [-1,1]
    t_indices = torch.randint(0, num_timesteps, (batch_size,))
    x_t, noise = scheduler.add_noise(x0, t_indices)
    assert x_t.shape == (batch_size, 1, 28, 28)
    assert noise.shape == (batch_size, 1, 28, 28)
    assert torch.isfinite(x_t).all()

    sqrt_ab = scheduler.alpha_bars[t_indices].sqrt().view(batch_size, 1, 1, 1)
    sqrt_omab = (1 - scheduler.alpha_bars[t_indices]).sqrt().view(batch_size, 1, 1, 1)
    assert torch.allclose(sqrt_ab**2 + sqrt_omab**2, torch.ones_like(sqrt_ab), atol=1e-6)
    assert torch.allclose(x_t, sqrt_ab * x0 + sqrt_omab * noise, atol=1e-6)

    # --- reverse (sampling) direction: one full ancestral pass, t=T-1..0 ---
    x = torch.randn(batch_size, 1, 28, 28)  # x_T ~ N(0,I)
    generator = torch.Generator().manual_seed(0)
    for t_index in reversed(range(num_timesteps)):
        t_batch = torch.full((batch_size,), t_index, dtype=torch.long)
        fake_eps_hat = torch.randn(batch_size, 1, 28, 28, generator=generator)  # stand-in for a real model
        x = scheduler.reverse_step(x, t_index, fake_eps_hat, generator=generator)
    assert x.shape == (batch_size, 1, 28, 28)
    assert torch.isfinite(x).all(), "reverse process produced non-finite values"

    # --- respaced schedule: defining invariant is EXACT alpha_bar match at kept t ---
    respaced = scheduler.make_respaced_schedule(num_inference_steps=10)
    assert respaced.num_inference_steps <= 10
    assert respaced.timestep_map[0].item() == 0
    assert respaced.timestep_map[-1].item() == num_timesteps - 1
    assert torch.allclose(respaced.alpha_bars, scheduler.alpha_bars[respaced.timestep_map], atol=1e-6), (
        "respacing must preserve the original marginal alpha_bar_t exactly at every kept timestep"
    )
    assert (respaced.betas > 0).all() and (respaced.betas < 1).all()

    x_short = torch.randn(batch_size, 1, 28, 28)
    for i in reversed(range(respaced.num_inference_steps)):
        fake_eps_hat = torch.randn(batch_size, 1, 28, 28, generator=generator)
        x_short = respaced.reverse_step(x_short, i, fake_eps_hat, generator=generator)
    assert torch.isfinite(x_short).all()

    print(f"ddpm_mnist_scheduler smoke test passed (T={num_timesteps}, respaced S={respaced.num_inference_steps}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run lightweight local checks and exit")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        raise SystemExit(
            "This file only defines the DDPM noise scheduler (use --smoke-test). "
            "See code/ddpm_mnist_train.py and code/ddpm_mnist_sample.py for actual use."
        )
