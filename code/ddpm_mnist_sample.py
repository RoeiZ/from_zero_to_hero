"""Sampling for the from-scratch MNIST DDPM: ancestral reverse sampling
(Ho et al., 2020, Algorithm 2), a trace of the x_T -> x_0 trajectory,
and a comparison of sampling with different numbers of reverse steps.
No training code and no architecture/schedule-math code lives here --
only the sampling LOOP that drives `code/ddpm_mnist_model.py` through
`code/ddpm_mnist_scheduler.py`.

--- x_T -> x_0 trace ---

`generate_samples` can snapshot x at a chosen set of timesteps while it
runs the reverse loop, so the whole denoising trajectory (pure noise ->
recognizable digit) can be displayed as one row per sample.

--- comparing different numbers of reverse (sampling) steps ---

A single TRAINED model (one fixed checkpoint, one fixed training
schedule) is sampled from using `DDPMScheduler.make_respaced_schedule`
at several different step counts S. This isolates ONE variable
(sampling-time compute budget) while holding everything else fixed:
same model weights, same training schedule, same initial x_T noise
(same seed) across every S compared -- so any visible difference in
the resulting digits is attributable to S alone, not to a confound
(per this project's research-integrity policy against multi-variable
causal claims).

No automated image-quality metric (e.g. FID) is computed here -- doing
that properly needs a reference feature extractor, which is out of
scope for this minimal exercise. The comparison below is deliberately
qualitative (visual grid) plus a measured, not interpreted, quantity:
wall-clock time and number of network evaluations (both exactly equal
to S, since this is the same model run S times). Any statement about
IMAGE QUALITY degrading with smaller S is visual/qualitative, not
backed by a metric -- see the printed discussion for the
observation/interpretation split.

Local usage (lightweight smoke test only, no checkpoint, no dataset):
    python code/ddpm_mnist_sample.py --smoke-test

Colab usage (real checkpoint from code/ddpm_mnist_train.py):
    python code/ddpm_mnist_sample.py \
        --checkpoint-path /content/drive/MyDrive/thesis_data/ddpm_mnist/checkpoint.pt \
        --num-samples 5 --step-counts 1000,250,50,10
"""

import argparse
import time

import torch
import matplotlib.pyplot as plt

from ddpm_mnist_model import MnistUNet
from ddpm_mnist_scheduler import DDPMScheduler, RespacedSchedule
from ddpm_mnist_train import load_checkpoint


def _schedule_steps(schedule) -> list[tuple[int, int]]:
    """Unify DDPMScheduler (T steps) and RespacedSchedule (S steps) into
    one list of (model_t, step_index) pairs, descending (x_T side first).
    model_t is the ORIGINAL 0-indexed timestep to feed the model;
    step_index is the index into `schedule`'s own (possibly short)
    coefficient tensors, for `schedule.reverse_step`."""
    if isinstance(schedule, DDPMScheduler):
        return [(t, t) for t in reversed(range(schedule.num_timesteps))]
    if isinstance(schedule, RespacedSchedule):
        return [
            (int(schedule.timestep_map[i].item()), i)
            for i in reversed(range(schedule.num_inference_steps))
        ]
    raise TypeError(f"unsupported schedule type: {type(schedule)}")


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    schedule,
    num_samples: int,
    image_shape: tuple[int, int, int],
    device: torch.device,
    generator: torch.Generator | None = None,
    track_model_timesteps: set[int] | None = None,
) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
    """Ancestral reverse sampling, paper Algorithm 2: start from
    x_T ~ N(0,I) and repeatedly apply the model + one reverse step until
    x_0 is reached.

    Args:
        model:    trained MnistUNet (or compatible), eps_hat = model(x_t, t)
        schedule: DDPMScheduler (full T steps) or RespacedSchedule (S steps)
        num_samples: how many independent images to generate at once
        image_shape: (C,H,W), e.g. (1,28,28)
        generator: torch.Generator already on `device`; if None, a fresh
            one is created (non-reproducible across calls)
        track_model_timesteps: optional set of ORIGINAL 0-indexed
            timesteps at which to snapshot x (for trajectory
            visualization); x_0 is always included as the final entry

    Returns:
        x0: [num_samples,C,H,W], the final generated images
        trajectory: list of (label, tensor[num_samples,C,H,W]) snapshots,
            in x_T -> x_0 order
    """
    model.eval()
    if generator is None:
        generator = torch.Generator(device=device)

    x = torch.randn(num_samples, *image_shape, generator=generator, device=device)  # x_T ~ N(0,I)

    trajectory: list[tuple[str, torch.Tensor]] = []
    for model_t, step_index in _schedule_steps(schedule):
        if track_model_timesteps is not None and model_t in track_model_timesteps:
            trajectory.append((f"t={model_t}", x.clone()))

        t_batch = torch.full((num_samples,), model_t, dtype=torch.long, device=device)  # [num_samples]
        eps_hat = model(x, t_batch)  # [num_samples,C,H,W]
        x = schedule.reverse_step(x, step_index, eps_hat, generator=generator)

    trajectory.append(("x0", x.clone()))
    return x, trajectory


def visualize_denoising_trajectory(
    trajectory: list[tuple[str, torch.Tensor]],
    num_examples: int = 5,
    save_path: str | None = None,
):
    """One row per example, one column per tracked timestep, x_T (left)
    to x_0 (right). Display only: each panel is independently min-max
    normalized to [0,1] for visibility (x_t at large t is ~N(0,1) noise,
    which has no meaningful fixed display range the way x0's [-1,1] does;
    this normalization is never applied to the underlying sampling math,
    only to what `imshow` renders)."""
    num_examples = min(num_examples, trajectory[0][1].shape[0])
    num_columns = len(trajectory)

    fig, axes = plt.subplots(num_examples, num_columns, figsize=(1.8 * num_columns, 1.8 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, num_columns)

    def to_displayable(image_1hw: torch.Tensor):
        image = image_1hw[0].detach().cpu()  # [1,H,W] -> [H,W]
        image_min, image_max = image.min(), image.max()
        return (image - image_min) / (image_max - image_min + 1e-8)

    for row in range(num_examples):
        for col, (label, snapshot) in enumerate(trajectory):
            axes[row, col].imshow(to_displayable(snapshot[row]), cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(label, fontsize=9)

    fig.suptitle("Reverse DDPM sampling trajectory: x_T (left) -> x_0 (right)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved trajectory figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def compare_num_inference_steps(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    step_counts: list[int],
    num_samples: int,
    image_shape: tuple[int, int, int],
    device: torch.device,
    seed: int,
) -> list[dict]:
    """Samples at each requested step count S, reseeding to the SAME
    seed (-> same x_T noise) before every S, so S is the only varied
    parameter. Returns one result dict per S with the generated images,
    wall-clock time, and the (trivial but explicit) number of model
    evaluations."""
    results = []
    for num_steps in step_counts:
        generator = torch.Generator(device=device).manual_seed(seed)

        if num_steps >= scheduler.num_timesteps:
            schedule = scheduler
            num_steps_used = scheduler.num_timesteps
        else:
            schedule = scheduler.make_respaced_schedule(num_steps)
            num_steps_used = schedule.num_inference_steps

        start_time = time.time()
        samples, _trajectory = generate_samples(
            model, schedule, num_samples, image_shape, device, generator=generator,
        )
        elapsed_seconds = time.time() - start_time

        results.append({
            "requested_steps": num_steps,
            "num_steps_used": num_steps_used,
            "samples": samples.cpu(),
            "elapsed_seconds": elapsed_seconds,
        })
    return results


def visualize_step_count_comparison(results: list[dict], save_path: str | None = None):
    """One row per step count S, one column per generated sample.
    Display: samples clamped from [-1,1] to [0,1] (same convention as
    code/ddpm_forward_process.py's `to_displayable`) -- x0 outputs
    SHOULD already be close to [-1,1] since that is the training
    pixel range, so clamping here only guards against small overshoot,
    unlike the trajectory plot's noisy intermediate steps."""
    num_rows = len(results)
    num_samples = results[0]["samples"].shape[0]

    fig, axes = plt.subplots(num_rows, num_samples, figsize=(1.8 * num_samples, 1.8 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, num_samples)

    for row, result in enumerate(results):
        samples = result["samples"]  # [num_samples,1,28,28], in approx [-1,1]
        for col in range(num_samples):
            image = ((samples[col, 0].clamp(-1, 1) + 1) / 2)  # [-1,1] -> [0,1]
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(
                    f"S={result['num_steps_used']}\n{result['elapsed_seconds']:.2f}s",
                    fontsize=9, rotation=0, labelpad=40, va="center",
                )
                axes[row, col].axis("on")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                for spine in axes[row, col].spines.values():
                    spine.set_visible(False)

    fig.suptitle("Same trained model, same initial noise -- varying only the number of reverse steps S")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved step-count comparison figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def print_step_count_discussion(results: list[dict]):
    """Prints the MEASURED quantities (time, steps) separately from the
    INTERPRETATION of what the visual grid is expected to show, per
    this project's research-integrity policy (observation vs.
    interpretation; no causal claim beyond the one controlled variable)."""
    print("\n--- measured ---")
    for result in results:
        print(
            f"S={result['num_steps_used']:>5}  "
            f"model evaluations={result['num_steps_used']:>5}  "
            f"wall_clock={result['elapsed_seconds']:.3f}s"
        )

    print(
        "\n--- interpretation (visual, not metric-backed) ---\n"
        "Each reverse step linearizes the true (near-continuous) reverse\n"
        "diffusion process into one Gaussian jump. With fewer steps S, each\n"
        "jump must undo a larger slice of the noise schedule at once, so the\n"
        "discretization error of that linear-Gaussian approximation grows.\n"
        "Expect: large S (e.g. the full training T) -> sharpest, most\n"
        "digit-like samples; small S (e.g. 10) -> blurrier, less coherent,\n"
        "sometimes digit-like-blob samples, despite using the EXACT same\n"
        "trained weights and the exact same starting noise. Judge this from\n"
        "the saved comparison figure, not from this printout alone."
    )


def run_smoke_test(device: torch.device):
    """Lightweight local verification: a freshly initialized (untrained)
    tiny model and a tiny-T scheduler, no checkpoint, no dataset. Checks
    generate_samples' output/trajectory shapes, the respaced-schedule
    sampling path, and that compare_num_inference_steps runs end-to-end
    and returns finite images for every requested step count."""
    torch.manual_seed(0)
    image_shape = (1, 28, 28)
    tiny_num_timesteps = 30

    model = MnistUNet(base_channels=16, time_embedding_dim=32).to(device)
    scheduler = DDPMScheduler(num_timesteps=tiny_num_timesteps, device=device)

    track_set = {tiny_num_timesteps - 1, 20, 10, 0}
    samples, trajectory = generate_samples(
        model, scheduler, num_samples=3, image_shape=image_shape, device=device,
        track_model_timesteps=track_set,
    )
    assert samples.shape == (3, 1, 28, 28)
    assert torch.isfinite(samples).all()
    assert len(trajectory) == len(track_set) + 1  # + the always-appended final x0
    assert trajectory[-1][0] == "x0"
    for _label, snapshot in trajectory:
        assert snapshot.shape == (3, 1, 28, 28)
    print(f"generate_samples (full schedule): output {tuple(samples.shape)}, "
          f"trajectory has {len(trajectory)} tracked snapshots")

    results = compare_num_inference_steps(
        model, scheduler, step_counts=[30, 10, 3], num_samples=2,
        image_shape=image_shape, device=device, seed=0,
    )
    assert len(results) == 3
    for result in results:
        assert result["samples"].shape == (2, 1, 28, 28)
        assert torch.isfinite(result["samples"]).all()
        assert result["num_steps_used"] <= tiny_num_timesteps
    print(f"compare_num_inference_steps: step counts used = {[r['num_steps_used'] for r in results]}")
    print_step_count_discussion(results)

    print("\nsmoke test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="run lightweight local verification with an untrained model, no checkpoint")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-trajectory-snapshots", type=int, default=8,
                         help="how many evenly-spaced timesteps to show in the x_T->x_0 trace")
    parser.add_argument("--step-counts", type=str, default="1000,250,50,10",
                         help="comma-separated list of S values to compare at sampling time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    assert args.checkpoint_path is not None, "--checkpoint-path is required outside --smoke-test"
    checkpoint = load_checkpoint(args.checkpoint_path, device)
    model = MnistUNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    scheduler = DDPMScheduler(**checkpoint["scheduler_config"], device=device)
    print(f"loaded checkpoint from {args.checkpoint_path} (epoch {checkpoint['epoch']})")

    image_shape = (checkpoint["model_config"]["in_channels"], 28, 28)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    track_model_timesteps = set(
        torch.linspace(0, scheduler.num_timesteps - 1, args.num_trajectory_snapshots).round().long().tolist()
    )
    _samples, trajectory = generate_samples(
        model, scheduler, args.num_samples, image_shape, device,
        generator=generator, track_model_timesteps=track_model_timesteps,
    )
    visualize_denoising_trajectory(
        trajectory, num_examples=args.num_samples,
        save_path=f"{args.output_dir}/ddpm_mnist_trajectory.png",
    )

    step_counts = [int(s) for s in args.step_counts.split(",")]
    results = compare_num_inference_steps(
        model, scheduler, step_counts, args.num_samples, image_shape, device, seed=args.seed,
    )
    visualize_step_count_comparison(results, save_path=f"{args.output_dir}/ddpm_mnist_step_count_comparison.png")
    print_step_count_discussion(results)


if __name__ == "__main__":
    main()
