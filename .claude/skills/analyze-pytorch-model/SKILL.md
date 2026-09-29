---
description: Analyze a PyTorch neural network without modifying it. Use
  to understand CNN, U-Net, Transformer, ViT or DiT architecture,
  forward flow, parameters, and tensor shapes.
name: analyze-pytorch-model
---

# Analyze PyTorch Model

Default to read-only analysis. 1. Open the actual model and relevant
helper modules. 2. Summarize the architecture. 3. Trace forward
input-to-output. 4. Give a table: operation, input shape, output shape,
purpose. 5. Identify trainable parameters. 6. Explain important
mathematical operations. 7. Identify down/up-sampling, residuals, skips,
concatenation, attention and conditioning. 8. Flag shape, broadcasting,
channel/token, dtype/device, gradient and normalization risks. For
U-Net, trace at least one skip connection. For Transformer/DiT, trace
`[B,N,D]`, Q/K/V, heads, and unpatchification when present. Do not run
full training. Use only tiny smoke tests consistent with `CLAUDE.md`.
