---
description: Review a research-code change or Git diff for mathematical
  correctness, tensor issues, unintended edits, reproducibility,
  gradient problems, and divergence from a paper. Use after model,
  diffusion, spectral, or experiment changes.
name: review-research-change
---

# Review Research Change

Review the actual diff and relevant surrounding code before editing
further.

Check: - mathematical fidelity and undocumented approximations - tensor
shapes, broadcasting, channels/tokens, dtype/device, complex handling -
autograd: detach, no_grad, in-place operations, missing gradients,
unregistered parameters - experiment validity: controls, seeds, config
logging, data split, checkpoints, metrics - scope: unrelated refactors,
unnecessary dependencies, hidden behavior changes, temporary files

Run/recommend only lightweight local checks; never full training
locally.

Report in severity order: 1. correctness blockers 2. research-validity
risks 3. maintainability concerns 4. verified-good aspects

For each issue name the file/function and smallest fix. Do not modify
unless requested.
