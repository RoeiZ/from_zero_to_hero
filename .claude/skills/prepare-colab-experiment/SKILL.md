---
description: Prepare PyTorch training, evaluation, diffusion sampling,
  or thesis experiments for Google Colab rather than the low-resource
  local PC. Use for GPU runs, checkpoints, Colab setup, experiment
  configs, and reproducibility.
name: prepare-colab-experiment
---

# Prepare Google Colab Experiment

The local PC is for development; Colab is for compute. Do not run the
full experiment locally.

## Readiness

Inspect entry script, config, dependencies, relative paths, device
handling, checkpoint/resume, output paths, and seeds. Flag
local/Windows-only assumptions.

## Local smoke test

When useful, run only tiny CPU verification: synthetic input or tiny
subset, at most 1-2 batches, forward, finite loss, backward, one
optimizer step, and shapes.

## Colab cells

Prepare minimal cells for: 1. GPU check 2. repository clone/pull 3.
dependency install 4. optional Google Drive mount 5. persistent
checkpoint/results path 6. exact training/evaluation command

Keep notebooks thin; algorithms remain in repository `.py` files.

## Persistence and reproducibility

Verify resume state for long runs. Important outputs must not exist only
in ephemeral `/content`. Record relevant Git commit, seed, config,
model, dataset/split, resolution, batch, optimizer/LR, diffusion/sampler
settings, GPU, runtime, peak memory, and metrics.

End with local checks completed, exact Colab commands, expected output,
persistent output path, resume command, and remaining risks.
