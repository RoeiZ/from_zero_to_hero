---
description: Convert a research-paper equation or algorithmic step into
  verified PyTorch code. Use for DDPM equations, diffusion schedules,
  spectral operations, reconstruction equations, and paper-faithful
  implementations.
name: implement-paper-equation
---

# Implement a Paper Equation

Do not jump directly to code. 1. Read the supplied source and identify
the equation/algorithm step. Do not invent unavailable details. 2.
Restate the equation and explain every symbol. 3. Create a table: math
symbol, code variable, shape, meaning. 4. Explain batch/spatial/timestep
indexing and broadcasting. 5. Define verification before integration:
shape, dtype/device, finite values, limiting cases, statistics, gradient
flow, fixed-seed behavior as applicable. 6. Implement the smallest clear
function; do not refactor unrelated code. 7. Run only lightweight local
checks. 8. Report passed/not-run checks and whether the implementation
is exact, equivalent, or approximate. If Colab execution is needed,
provide the exact next command/cell.
