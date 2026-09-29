---
description: Trace and debug PyTorch tensor shapes, broadcasting,
  reshaping, channels, tokens, attention dimensions, timestep
  embeddings, dtype, and device compatibility.
name: verify-tensors
---

# Verify Tensor Shapes

Inspect actual code before conclusions. Do not modify architecture
initially. For each important tensor report name, meaning, shape, and
relevant dtype/device. Pay special attention to `view`, `reshape`,
`flatten`, `unsqueeze`, `squeeze`, `permute`, `transpose`, `cat`,
broadcasting, convolutions, patchification, attention heads, timestep
embeddings, and complex real/imaginary representations. For broadcasting
show original shapes, inserted singleton dimensions, final broadcast
shape, and why it is mathematically correct. If an error exists report
expected shape, actual shape, first divergence, root cause, and smallest
correction. Do not implement unless requested. Use tiny synthetic
tensors only; never launch training for shape debugging.
