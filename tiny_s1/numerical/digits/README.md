# Optical-digits numerical comparator

This source compares two algebraically equivalent evaluations of the sigmoid
inside a parity BCE backward pass. The forward BCE-with-logits expression is
identical. A shared `64 -> 64 ReLU -> 32 ReLU` MLP also predicts ten digit classes.
All training floating arrays, reductions, gradients and Adam states use FP32.
Evaluation uses FP32 forward logits with a common stable float64 scoring routine.

`protocol.json` specifies the data hashes, training configuration, run list and
validation tolerances. It is hashed into the preflight gate and source freeze.
The numerical code is derived from a separately executed, frozen implementation;
`provenance.json` distinguishes the original and portable source hashes.

Use Python 3.12 and the dependencies pinned in
[`../requirements-numerical.txt`](../requirements-numerical.txt). From the
repository root:

```sh
python -m pip install -r tiny_s1/numerical/requirements-numerical.txt
python tiny_s1/numerical/digits/fetch_data.py --data-dir ../digits-data
python tiny_s1/numerical/digits/validate.py --data-dir ../digits-data --out ../digits-runs/preflight.json
python tiny_s1/numerical/digits/train.py freeze --data-dir ../digits-data --out-dir ../digits-runs --gate ../digits-runs/preflight.json
python tiny_s1/numerical/digits/train.py run --data-dir ../digits-data --out-dir ../digits-runs --gate ../digits-runs/preflight.json
```

Defaults place data, checkpoints and reports in `digits-numerical-output`, a
sibling of the repository. Explicit paths are recommended. An existing freeze or
run directory is not overwritten. To verify existing data without network access
or writes, add `--verify-only` to the downloader command.

The preflight opens training data only. It checks initial and 50-update states
against independently implemented Torch losses/gradients, identical-gradient
Adam steps, 80-digit scalar calculations, and 20 exact repeated updates. It must
pass before `freeze` or `run` will proceed. Source/protocol changes invalidate the
gate. A freeze also verifies every official data-file hash. It copies executable
source and `protocol.json` into the output folder.

`run` performs six shared runs (seeds 0, 1, 2, each with both expressions) plus a
detached-parity seed-0 pair. Each uses 50 epochs, 1,500 updates, batch size 128
including its final partial batch. Initialization uses `default_rng(seed)` and
permutations use `default_rng(1000 + seed)`, copied within each pair. Checkpoints
include parameters, Adam moments and counters. NLL, Brier, accuracy, probability
TV and action disagreements are recorded, alongside cross-seed comparisons.
Three seeds support descriptive comparisons; test examples are not independent
training replications. Matching across different hardware/libraries is not assumed.

## Dataset

The [official UCI Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
is by E. Alpaydin and C. Kaynak (1998), DOI
[10.24432/C50P49](https://doi.org/10.24432/C50P49), under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The downloader records
attribution, license, original byte hashes, parsed-array hashes and class counts.
It preserves `optdigits.tra` (3,823 examples) and `optdigits.tes` (1,797 examples),
the original split with different training/test writers. Pixels are divided by
16 and parity labels are derived during loading; raw files remain unchanged.
`sklearn.load_digits()` contains only the original test examples and is not used.

Raw data, generated checkpoints, predictions and research outcomes are written
outside this source directory.
