# Master's Thesis --- Claude Code Project Instructions

## Project Goal

This repository supports a Master's thesis on diffusion models for image
reconstruction. The path includes PyTorch, CNNs, autoencoders, U-Net,
denoising, DDPM/DDIM, Transformers, ViT, DiT, inverse problems,
Fourier/spectral analysis, Spectral Progressive Diffusion, reproduction,
ablations, and thesis experiments.

Treat the user as the researcher/architect and Claude as the
implementation partner. Prefer readable research code over clever code.
Connect important code to mathematics and always make important tensor
shapes explicit.

## Working Method

For substantial tasks: 1. Inspect relevant files before making claims.
2. Explain current behavior when needed. 3. Define one conceptual change
at a time. 4. Plan before editing when the change is non-trivial or
requirements are ambiguous. 5. Implement only the requested change; do
not refactor unrelated code. 6. Run lightweight local verification when
safe. 7. Review the diff. 8. Explain the mathematical and tensor-shape
meaning of the change. 9. State what still needs to be executed in
Google Colab.

Avoid over-engineering.

## Code Style

Use Python and PyTorch. Prefer explicit code, descriptive names, small
functions, simple module boundaries, configuration-driven experiments,
and comments for mathematically non-obvious operations. Avoid clever
one-liners, unnecessary abstractions, hidden global state, unnecessary
libraries, silent algorithm changes, and local absolute paths.

## Data split and validation
When writing the code please make sure you split the data into training set and testing set. Use explicit and clear name for the variables represent the splitting. 

## Tensor Shape Rule

Tensor shapes are first-class information.

Common conventions: - images: `[B,C,H,W]` - timesteps: `[B]` -
Transformer tokens: `[B,N,D]` - attention: `[B,heads,N,N]` - time
embedding: `[B,D]` before spatial broadcasting

For
reshape/view/flatten/unsqueeze/squeeze/permute/transpose/cat/patchification/broadcasting,
explain shape before, shape after, and why the operation is required.

## Mathematical Implementations

When implementing a paper equation: 1. Identify/restates the equation or
algorithm step. 2. Explain every symbol. 3. Map symbols to code
variables. 4. State tensor shapes. 5. Explain indexing/broadcasting. 6.
Implement the smallest clear version. 7. Add a focused verification
test. 8. State whether it is exact, mathematically equivalent, or
approximate.

If implementation differs from the paper, explicitly state what differs,
why, expected consequences, and whether it affects reproduction claims.
Never silently replace a paper method with a convenient approximation.

## Research Integrity

Separate observations, measurements, interpretations, and hypotheses. Do
not make causal claims when multiple uncontrolled variables changed. For
reproduction, document settings that cannot be matched. Do not invent
paper details; inspect the actual source when needed.

# Compute Environment

## Local Computer

This project is developed locally using Cursor and Claude Code. The
local computer has limited computational resources. It is the
DEVELOPMENT environment, not the training machine.

Allowed locally: - editing and repository inspection - code review and
Git - syntax/import checks when lightweight - tiny unit tests - tiny
synthetic tensors - CPU smoke tests - tensor-shape verification - one or
two tiny forward/backward passes when clearly lightweight

Do NOT run locally by default: - full model training - long training
loops - large datasets/batches - expensive diffusion sampling - GPU
benchmarks - full evaluation suites - large image/video generation -
hyperparameter sweeps - memory-intensive experiments

If uncertain whether a command is expensive, do not run it locally.
Prepare it for Colab instead.

## Google Colab

Google Colab is the primary COMPUTE/TRAINING environment.

Use Colab for: - GPU training - full evaluation - diffusion sampling -
large datasets - benchmarks - thesis experiments - checkpoint
generation - runtime/GPU-memory measurement - expensive spectral
analysis

### Dataset Caching (download once, not every run)

Colab's local disk (`/content/...`) is ephemeral and wiped whenever the
runtime recycles or disconnects, so a dataset downloaded there is
re-downloaded on every fresh session. Instead, mount Google Drive and
point the dataset's root/download directory at a persistent Drive path
(configurable, not hard-coded). Dataset loaders should check whether the
data already exists at that path and only download if missing.

``` python
from google.colab import drive
drive.mount('/content/drive')

dataset_root = '/content/drive/MyDrive/thesis_data/cifar10'
train_set = torchvision.datasets.CIFAR10(
    root=dataset_root, train=True, download=True, transform=transform
)
```

Tradeoff: reading many small files directly from Drive can be slower
than local disk. For small datasets (e.g. CIFAR-10) this is negligible;
for larger datasets, prefer copying from Drive to local `/content` once
at the start of each run, then reading locally for speed.

All training/experiment code must work in a fresh Colab runtime. Never
assume the local PC has CUDA.

## Device Handling

Do not hard-code CUDA unless explicitly required. Prefer:

``` python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Colab Compatibility

Avoid Windows-only absolute paths, local-only environment variables, and
local-only datasets. Prefer repository-relative paths and configurable
persistent-storage paths.

## Recommended Repository Structure

``` text
master-thesis/
├── CLAUDE.md
├── .claude/
│   └── skills/
├── README.md
├── requirements.txt
├── configs/
├── code/
├── notebooks/
│   └── colab/
├── src/
│   ├── models/
│   ├── diffusion/
│   ├── reconstruction/
│   ├── spectral/
│   └── utils/
├── scripts/
├── tests/
├── experiments/
└── papers/
```

Create directories only when needed.
wite notebooks for Colan and also .py files and saves the .py files under code with appropiate name.

## Colab Notebook Philosophy

Colab notebooks should be thin orchestration layers: 1. check GPU 2.
clone/pull repository 3. install dependencies 4. optionally mount Google
Drive 5. choose configuration 6. launch training/evaluation scripts 7.
display selected results

Core algorithms belong in version-controlled `.py` files, not only in
notebooks.

## Git / Colab Workflow

Preferred flow:
`Cursor + Claude Code -> lightweight verification -> Git commit/push -> Colab clone/pull -> GPU run -> persistent checkpoints/results`.

Do not rely on manually copying source code between Cursor and Colab.

## Smoke Tests

Important components should support lightweight verification with tiny
tensors or a tiny data subset. Verify as applicable: imports,
construction, forward pass, output shape, finite loss, backward pass,
expected gradients, one optimizer step, and device compatibility.

A smoke test is NOT evidence that the model trains correctly or
reproduces a paper.

## Training Scripts

Prefer Colab launching commands such as:

``` bash
python train.py --config configs/ddpm.yaml
```

Keep relevant dataset, resolution, batch size, learning rate,
epochs/steps, diffusion steps, sampler, inference steps, seed,
checkpoint path, and output path configurable.

## Checkpoints

Colab sessions can terminate. Long runs should support save/resume.
Checkpoints should normally contain model state, optimizer state,
epoch/global step, experiment configuration, scheduler state if used,
and other state required for faithful resume.

Do not keep the only important checkpoint in ephemeral Colab storage.
Persistent paths such as Google Drive should be configurable.

## Experiment Logging

For thesis experiments record the relevant: experiment ID, Git commit,
seed, model/config, dataset/split, resolution, batch size, optimizer/LR,
training steps, diffusion schedule, sampler, inference steps, GPU,
runtime, peak memory, metrics, and output/checkpoint paths. Prefer
machine-readable logs.

## Verification Policy

Never claim code is correct merely because it looks plausible. Use the
strongest practical check: unit test, shape test, numerical sanity
check, limiting case, gradient check, smoke test, equation comparison,
or paper/official-code comparison.

After changes summarize: - files changed - conceptual change - checks
run - checks not run - what must be tested in Colab

## Results Visualization

Every exercise/project must include a qualitative results-visualization
step, not only scalar metrics. After evaluation, show a small number of
concrete examples (around 5) with: input, ground-truth label/target,
and model output/prediction, displayed together (e.g. image grid with
titles showing true vs. predicted, or input/reconstruction/target
side-by-side for reconstruction and diffusion tasks).

Prefer visualizing as much as is informative for the task, for example:
- sample predictions vs. ground truth (classification)
- input / reconstruction / target triplets (autoencoders, inverse
  problems)
- noising/denoising steps or generated samples (DDPM/DDIM)
- loss curves and any other relevant training curves
- attention maps or spectral plots when relevant to the method

Keep visualization code in the `.py` script/module (e.g. a
`visualize_predictions`-style function), with the Colab notebook simply
calling it and displaying the output, consistent with the
"notebooks are thin orchestration layers" philosophy.

## Claude Execution Policy

Classify meaningful execution as: A. lightweight local verification B.
Colab training/experiment

A may run locally when safe. B must not run locally by default. For B:
prepare code/config, run only useful smoke tests locally, provide exact
Colab command/cells, use persistent output storage when needed, and
support resume for long training.

## Git Safety

Inspect Git status before broad edits. Never discard user changes,
force-push, destructively reset history, delete important outputs, or
commit secrets without explicit approval.

## Skills

Project procedures live under `.claude/skills/`. Core skills: -
`analyze-pytorch-model` - `implement-paper-equation` -
`verify-tensors` - `prepare-colab-experiment` - `review-research-change`

## Autonomy

For local, reversible development actions inside this repository,
proceed without asking for confirmation.

This includes:
- reading project files
- editing project files
- creating normal source/test files
- running lightweight Python scripts
- running unit tests and smoke tests
- running shel commands that are part of writing the code.
- git status
- git diff
- git log

Do not repeatedly ask for confirmation for routine,
reversible development work.

Ask before:
- git push
- deleting important files
- destructive Git operations
- installing system-level software
- modifying files outside this repository
- running computationally expensive operations
- full model training

Full training must be prepared for Google Colab rather than
executed on the local computer.