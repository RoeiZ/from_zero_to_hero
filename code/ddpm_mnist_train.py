"""Training loop for a minimal from-scratch DDPM on MNIST. No diffusers
-- this file only wires together the model (code/ddpm_mnist_model.py)
and the noise scheduler (code/ddpm_mnist_scheduler.py); it contains no
architecture code and no schedule-math code of its own.

Training objective (Ho et al., 2020, Eq. 14, the "simple" loss):

    L_simple(theta) = E_{x0, t, eps} [ || eps - eps_theta(x_t, t) ||^2 ]

where x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps (the
scheduler's `add_noise`) and eps_theta = model (eps_hat). t ~
Uniform{0,...,T-1} and eps ~ N(0,I) are both freshly resampled for
EVERY batch, so this is a Monte-Carlo estimate of the loss, not an
exact value -- train_loss/test_loss below both inherit that sampling
noise (documented again at `evaluate`).

Train/test split: torchvision's native MNIST train=True/False
partition (60000 / 10000 images), per this project's data-splitting
policy. The test split here is NOT used for early stopping or model
selection (this is a generative model, not a discriminative one) --
it is used only to monitor whether the denoising MSE objective
generalizes to held-out images, as an observational diagnostic.

Local usage (lightweight smoke test only, no dataset download):
    python code/ddpm_mnist_train.py --smoke-test

Colab usage (real data, full training, checkpoint to Drive):
    python code/ddpm_mnist_train.py \
        --data-root /content/drive/MyDrive/thesis_data/mnist \
        --checkpoint-path /content/drive/MyDrive/thesis_data/ddpm_mnist/checkpoint.pt \
        --epochs 20 --batch-size 128 --lr 2e-4 --num-timesteps 1000
"""

import argparse
import os

import torch
import torch.nn as nn

from ddpm_mnist_model import MnistUNet, count_trainable_parameters
from ddpm_mnist_scheduler import DDPMScheduler


def get_mnist_dataloaders(data_root: str, batch_size: int):
    """Real MNIST train/test split (torchvision's native partition),
    pixels rescaled from [0,1] (ToTensor) to [-1,1] (DDPM convention,
    matching code/ddpm_forward_process.py). Downloads ~12MB on first
    call. Intended for Colab, not local smoke tests.
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),                  # [0,255] uint8 HxW -> [0,1] float32 [1,28,28]
        transforms.Normalize((0.5,), (0.5,)),    # [0,1] -> [-1,1]: (x-0.5)/0.5
    ])

    mnist_train_dataset = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform,
    )
    mnist_test_dataset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass over `loader`. For each batch: draw a random t per
    image, noise the batch to x_t, predict eps_hat, minimize
    MSE(eps_hat, eps). Returns the batch-size-weighted mean loss."""
    model.train()
    loss_fn = nn.MSELoss()
    total_loss, total_samples = 0.0, 0

    for images, _labels in loader:
        x0 = images.to(device)  # [B,1,28,28], in [-1,1]; label unused (unconditional DDPM)
        batch_size = x0.shape[0]

        t_indices = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)  # [B]
        x_t, noise = scheduler.add_noise(x0, t_indices)  # [B,1,28,28], [B,1,28,28]

        optimizer.zero_grad()
        eps_hat = model(x_t, t_indices)        # [B,1,28,28]
        loss = loss_fn(eps_hat, noise)         # scalar, L_simple Monte-Carlo estimate
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    scheduler: DDPMScheduler,
    loader,
    device: torch.device,
) -> float:
    """Held-out denoising MSE, same objective as training but on the
    test split and with no gradient step. t and eps are still randomly
    resampled each call, so this is an approximate monitoring signal
    (expect run-to-run jitter at fixed model weights), not a
    deterministic metric."""
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss, total_samples = 0.0, 0

    for images, _labels in loader:
        x0 = images.to(device)
        batch_size = x0.shape[0]

        t_indices = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        x_t, noise = scheduler.add_noise(x0, t_indices)
        eps_hat = model(x_t, t_indices)
        loss = loss_fn(eps_hat, noise)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: dict,
    scheduler_config: dict,
    train_loss_history: list,
    test_loss_history: list,
    seed: int,
):
    """Checkpoint contents follow this project's checkpointing policy:
    model state, optimizer state, epoch, and the configs needed to
    reconstruct the model and scheduler exactly (so `load_checkpoint`
    never has to guess an architecture/schedule hyperparameter)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "model_config": model_config,
        "scheduler_config": scheduler_config,
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
        "seed": seed,
    }, path)
    print(f"saved checkpoint to {path} (epoch {epoch})")


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Returns the raw checkpoint dict; caller reconstructs model/scheduler
    from `model_config`/`scheduler_config` (see code/ddpm_mnist_sample.py)."""
    return torch.load(path, map_location=device, weights_only=False)


def run_smoke_test(device: torch.device):
    """Lightweight local verification: synthetic tensors only, NO MNIST
    download. Builds a tiny model + tiny-T scheduler, runs a few manual
    training steps on random batches (not a real DataLoader/dataset),
    checks the loss is finite and decreases on a fixed synthetic batch
    with enough steps, and round-trips a checkpoint save/load."""
    torch.manual_seed(0)

    tiny_num_timesteps = 50
    model = MnistUNet(base_channels=16, time_embedding_dim=32).to(device)
    scheduler = DDPMScheduler(num_timesteps=tiny_num_timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    # Fixed synthetic "dataset": one batch, reused every step, so a
    # decreasing loss is attributable to optimization, not changing data.
    batch_size = 8
    x0 = torch.rand(batch_size, 1, 28, 28, device=device) * 2 - 1  # [B,1,28,28], in [-1,1]

    losses = []
    for _step in range(60):
        t_indices = torch.randint(0, tiny_num_timesteps, (batch_size,), device=device)
        x_t, noise = scheduler.add_noise(x0, t_indices)
        optimizer.zero_grad()
        eps_hat = model(x_t, t_indices)
        loss = loss_fn(eps_hat, noise)
        assert torch.isfinite(loss), "loss is not finite"
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"loss trace (every 10th step): {[f'{l:.3f}' for l in losses[::10]]}")
    # t and noise are re-sampled every step, so this is noisy -- compare
    # mean of the first 10 vs. last 10 steps rather than single points.
    early_mean = sum(losses[:10]) / 10
    late_mean = sum(losses[-10:]) / 10
    assert late_mean < early_mean, (
        f"mean loss did not decrease (early={early_mean:.4f}, late={late_mean:.4f}); optimization is not working"
    )
    print(f"mean loss decreased: early={early_mean:.4f} -> late={late_mean:.4f} (optimizer is learning)")

    # --- checkpoint save/load roundtrip ---
    scratch_checkpoint_path = os.path.join(
        os.environ.get("TEMP", "."), "ddpm_mnist_smoke_checkpoint.pt"
    )
    model_config = dict(in_channels=1, out_channels=1, base_channels=16, time_embedding_dim=32)
    save_checkpoint(
        scratch_checkpoint_path, model, optimizer, epoch=5,
        model_config=model_config, scheduler_config=scheduler.config(),
        train_loss_history=losses, test_loss_history=[], seed=0,
    )
    checkpoint = load_checkpoint(scratch_checkpoint_path, device)
    reloaded_model = MnistUNet(**checkpoint["model_config"]).to(device)
    reloaded_model.load_state_dict(checkpoint["model_state_dict"])
    for p_orig, p_reloaded in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.equal(p_orig, p_reloaded), "reloaded weights do not match saved weights"
    assert checkpoint["epoch"] == 5
    assert checkpoint["scheduler_config"]["num_timesteps"] == tiny_num_timesteps
    print(f"checkpoint save/load roundtrip ok ({scratch_checkpoint_path})")
    os.remove(scratch_checkpoint_path)

    print("\nsmoke test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="run lightweight local verification with synthetic tensors, no download")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/ddpm_mnist.pt")
    parser.add_argument("--resume", action="store_true", help="resume from --checkpoint-path if it exists")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--time-embedding-dim", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=1, help="save a checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    torch.manual_seed(args.seed)

    model_config = dict(
        in_channels=1, out_channels=1,
        base_channels=args.base_channels, time_embedding_dim=args.time_embedding_dim,
    )
    model = MnistUNet(**model_config).to(device)
    scheduler = DDPMScheduler(args.num_timesteps, args.beta_start, args.beta_end, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    start_epoch = 0
    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    if args.resume and os.path.exists(args.checkpoint_path):
        checkpoint = load_checkpoint(args.checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        train_loss_history = checkpoint["train_loss_history"]
        test_loss_history = checkpoint["test_loss_history"]
        print(f"resumed from {args.checkpoint_path} at epoch {start_epoch}")

    train_loader, test_loader = get_mnist_dataloaders(args.data_root, args.batch_size)

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, scheduler, train_loader, optimizer, device)
        test_loss = evaluate(model, scheduler, test_loader, device)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_mse={train_loss:.4f}  test_mse={test_loss:.4f}")

        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(
                args.checkpoint_path, model, optimizer, epoch=epoch + 1,
                model_config=model_config, scheduler_config=scheduler.config(),
                train_loss_history=train_loss_history, test_loss_history=test_loss_history,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
