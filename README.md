# TinyS1

A 116,545-parameter NumPy transformer for probabilistic decisions on synthetic
support tickets. It implements caller-supplied Choice options, ordered Score
levels and binary Noul questions. Autodiff, attention, loss functions and AdamW
are implemented locally.

This is an educational model with a fixed word vocabulary. It is not TypeSafe's
Jev, an implementation of RLCD, or a general-purpose language model.

## Setup and checks

Tested with Python 3.12.14, NumPy 2.5.4 and Matplotlib 3.11.2.

```bash
python -m venv .venv
# Activate .venv in your shell.
python -m pip install -r requirements.txt
cd tiny_s1
python tests.py
python test_regressions.py
python validate_artifacts.py
python api.py
python structural_probe.py
```

`api.py` accepts `--request path/to/request.json` or runs its built-in example.
It is a local adapter, not an HTTP server. State length is limited to 39 word
tokens, question blocks to 23 tokens and option descriptions to nine words.
Choice accepts 1-255 options; Score accepts 2-10 levels. Unknown words become
`[UNK]`. Dense attention limits practical request sizes.

## Re-evaluate saved checkpoints

Set one BLAS thread before starting Python.

```powershell
# PowerShell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
```

```bash
# Unix shells
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

From `tiny_s1`:

```bash
python evaluate.py --checkpoint-dir ../artifacts/supplied --output-dir ../results/reproduced
python evaluate.py --checkpoint-dir ../artifacts/fresh --output-dir ../results/fresh
```

The validation split has 2,000 tickets; the ordinary test split has 4,000.
Escape-option conditions each have 2,000 tickets. Temperature fitting uses only
the validation split. Seeds and checkpoint hashes are recorded with results.

## Train

```bash
python train.py --run stream --steps 4000 --batch 32 --seed 1 --output-dir ../artifacts/runs/seed-1
python train.py --run fixed --steps 4000 --batch 32 --seed 1 --output-dir ../artifacts/runs/seed-1
python train.py --run other --steps 4000 --batch 32 --seed 1 --output-dir ../artifacts/runs/seed-1
```

`stream` samples fresh synthetic tickets. `fixed` resamples a 512-ticket pool.
`other` offers an escape option on 30% of training requests, half of which have
no correct listed team. A short execution check can use `--steps 25` with a
separate output directory.

## Lexical baselines

```bash
python baselines.py --train-requests 128000 --seeds 0 1 2 3 4
```

This counts exact matches between state words and each option description. It
fits a shared logit scale, or a two-parameter scale conditioned on option count,
using sampled Choice outcomes. Bayes probabilities are not used for fitting.
Both variants reach 74.175% on the ordinary test set and 73.700% on the held-out
vocabulary set for each of the five training seeds. The option-count variant has
mean test log loss 0.66481, versus 0.69920 for the fixed-scale variant.

These are handcrafted Choice-only baselines for the cue-word generator; they do
not implement Score, Noul or general language understanding. The fixed-scale
variant preserves relative odds between shared options, showing that this
restriction alone does not require low classification accuracy on this task.
Raw results and fit diagnostics are in `results/lexical-baselines.json`.

## Recorded results and limitations

The recorded full runs use seed 0, 4,000 steps and batch 32.

| Regime | Supplied checkpoint accuracy | New training accuracy |
|---|---:|---:|
| stream | 66.650% | 50.925% |
| fixed | 33.275% | 32.550% |
| other | 70.225% | 49.850% |

These are Choice accuracies on the same synthetic test split. Re-evaluating the
supplied weights reproduces their accuracy; new training does not reproduce the
stronger stream/other results. The cause is unresolved. One seed per regime is
insufficient to characterize training variability.

`artifacts/` contains weights and training logs; `results/` contains machine-readable
evaluations and checks. `results/summary.csv` compares the recorded conditions.
The synthetic oracle is a reference for this generator, not for real text.
See `bayes_other.py` for the full-information escape-option oracle and
`structural_probe.py` for the shared-log-odds restriction of isolated option scoring.

Choice ECE bins top probability. Noul ECE bins P(true). Choice Brier sums squared
errors across classes; Noul Brier uses one binary squared error. The custom
`confidence` output measures entropy concentration and is not a calibrated
correctness probability. Option-order symmetry and question isolation arise from
this implementation's mask and fixed position offsets.

MIT licensed.
