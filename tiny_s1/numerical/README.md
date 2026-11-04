# Numerical diagnostics

These scripts import the existing core from their parent `tiny_s1` directory.
They run from any working directory. Choose output paths outside the repository
to keep generated checkpoints, predictions and reports separate from source.
The NumPy model, data generator, autograd and optimizer core are unchanged.

## Environment

The recorded Windows CPU environment used Python 3.12.14, NumPy 2.5.4 and
PyTorch 2.15.0+cpu. Install the numerical dependencies in a fresh environment:

```sh
python -m pip install -r tiny_s1/numerical/requirements-numerical.txt
```

The requirements use the official PyTorch CPU wheel index. For another supported
platform, consult the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
and select CPU; keep the recorded version when reproducing this environment.
The pins record the tested environment; wheel availability depends on platform
and Python version. The scripts set BLAS/OpenMP threads to one before NumPy
imports. The independent checker also sets one Torch thread and deterministic
algorithms. Bitwise identity across different libraries/hardware is not assumed.

## Paired training and saved predictions

Run both expressions for each seed `0`, `1` and `2`, retaining every run. For
example, from the repository root:

```sh
python tiny_s1/numerical/pilot_train.py --seed 0 --formula stable --steps 4000 --output ../numerical-runs/runs/seed-0/stable
python tiny_s1/numerical/pilot_train.py --seed 0 --formula original --steps 4000 --output ../numerical-runs/runs/seed-0/original
```

`original` denotes the direct reciprocal sigmoid expression. Only that BCE
backward expression differs. Each run records source, initial-parameter and
training-batch hashes, optimizer moments, final weights and prediction arrays.
The canonical configuration uses batch 32, FP32 parameters and FP64 sampled
targets, yielding mixed backward dtypes. Training data use seed `1000 + seed`.

Evaluation defaults are validation seed 5001 / 2,000 requests and test seed 5002 /
4,000 requests. Explicit overrides are `--validation-seed`,
`--validation-requests`, `--test-seed`, and `--test-requests`. The fixed pilot
analysis below requires the default evaluation sets and 4,000 updates. For a
two-update source check without generating any held-out data, add
`--steps 2 --skip-evaluation`; such outputs are not inputs to the pilot analysis.
Short pilot runs use the core schedule for their requested total; the first 200
warmup updates agree with a 4,000-update run.

## Matched arithmetic traces

```sh
python tiny_s1/numerical/trace_pairs.py --mode canonical_fp32 --seed 0 --steps 200 --batch 32 --output-dir ../numerical-traces/canonical
```

Other modes are `typed_targets_fp32`, `fp64_from_fp32`, and
`detached_bce_fp32`. FP64 starts from exactly FP32-representable initial arrays.
Detachment blocks only the binary loss route into the shared representation.
Every trace uses `lr_at(step, 4000)`, regardless of trace length. Same-state
counterfactual gradients are kept separate from diverged-trajectory comparisons.
The trace script generates training batches only; it does not evaluate a test
set. Its scalar FP64 comparison is a floating-point reference, not exact
arithmetic or a high-precision proof for the FP64 condition.

## Independent framework checks

```sh
python tiny_s1/numerical/framework_reference.py --output ../numerical-checks/initial.json
python tiny_s1/numerical/framework_reference.py --stable-checkpoint ../numerical-runs/runs/seed-0/stable/checkpoints/ckpt_stream.npz --direct-checkpoint ../numerical-runs/runs/seed-0/original/checkpoints/ckpt_stream.npz --output ../numerical-checks/trained.json
```

Without checkpoint arguments, initial-state and controlled-gradient optimizer
checks run, and `trained_state_coverage.status` explicitly reports
`not_requested`. Either or both explicit checkpoints may be supplied. Trained
checks use fresh diagnostic batches and do not evaluate generalization.

Loss/gradient comparison tolerances are FP64 `atol=1e-10, rtol=1e-9` and FP32
`atol=3e-6, rtol=3e-5`. Controlled optimizer tolerances are FP64
`1e-12, 1e-10` and FP32 `2e-7, 2e-5`. The reported block criterion is
`max_abs_error <= atol + rtol * max_abs_scale`. Near-zero key-bias gradients
require absolute interpretation. Independent multi-update parameter drift is
reported, not used as a cross-framework bitwise-equivalence assertion.

## Fixed pilot analysis

After all six 4,000-update runs and their predictions exist:

```sh
python tiny_s1/numerical/analyze_pilot.py --data-root ../numerical-runs --output-dir ../numerical-analysis
```

The analysis validates saved source and paired data metadata, uses a common FP64
probability evaluation, and writes JSON plus per-seed CSV metrics. Binary Brier
uses one Bernoulli coordinate; multiclass Brier sums category errors. Raw
probabilities are retained; legacy binary metric checks allow only the documented
clipping-scale difference. ECE and the fixed 0.8 Choice acceptance example are
secondary diagnostics. Three seeds and three correlated between-seed pairs per
expression support descriptive summaries, not confirmation or a deployment-risk
guarantee. `provenance.json` distinguishes the executed work scripts from these
portable adaptations; these are not identical source artifacts.
