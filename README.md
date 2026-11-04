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
stronger stream/other results. The stream discrepancy is partly diagnosed by the common-environment controls
below. The supplied archive is still not recovered exactly. One seed per
multitask regime is insufficient to characterize training variability.

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

## Numerical comparison tools

Optional tools in [`tiny_s1/numerical`](tiny_s1/numerical/README.md) provide
matched sigmoid-gradient training, same-state traces, an independent PyTorch
reference, and analysis of saved predictions. The
[`digits` comparator](tiny_s1/numerical/digits/README.md) uses a separate shared
MLP and the original UCI train/test split. Install their separate numerical
requirements and choose output directories outside the source repository.

The existing Transformer core has FP32 parameters and forward values, but its
FP64 targets yield mixed backward precision: 24 FP64 and eight FP32 gradient
arrays in the audited multitask batch. Optimizer moments remain FP32. Equivalent
gradient expressions are separate numerical training conditions; these tools do
not assert a general accuracy advantage for either expression. Their provenance
files distinguish frozen executed sources from portable adaptations.

## Controlled option-context experiments

The fixed protocol in `experiments/option_context_v1.json` compares isolated
options, explicit option count, independent random context and context pooled
through each question. It uses two cue-noise models and three seeds per condition.
Three additional runs compare the provided and current multitask code and a
single-expression numerical control in the same environment.
`experiments/reference_numpy/` preserves only the four
provided training source files for that comparison.

The recorded NumPy version can be installed from the repository root with
`python -m pip install -r experiments/requirements_option_context_v1.txt`.

```bash
cd tiny_s1
python test_context_study.py
python run_context_study.py --protocol ../experiments/option_context_v1.json --output-dir ../artifacts/context-study-v1 --workers 3
```

This runs 24 Choice-only models and three multitask reproduction controls, each for
4,000 steps. It records source, data and weight hashes and retains all conditions.
The default generator remains unchanged. Training is local and requires no API
key. Use `--dry-run` to inspect the execution list without training.

After all jobs finish, the supplementary lexical fits and aggregate checks use
the same protocol and request streams:

```bash
python context_baselines.py --protocol ../experiments/option_context_v1.json --output ../artifacts/context-study-v1/lexical-controls.json
python verify_training_streams.py --output ../artifacts/context-study-v1/training-stream-replay.json
python analyze_context_study.py --study-dir ../artifacts/context-study-v1
```

The analyzer checks data and initialization matches, checkpoint hashes,
permutation checks and metrics recomputed from saved predictions. It reports
all seeds, raw results, validation-only temperature scaling and paired
comparisons. Three-seed standard deviations describe training variability;
they are distinct from uncertainty over test examples.
Source-file hashes are always recorded. A clean Git checkout also records its
commit; `EXPERIMENT_SOURCE_COMMIT` can supply that identifier explicitly.

## Completed option-context study

All 24 registered Choice-only runs and three multitask reproduction controls
completed. Training source is commit
`c4e1ad0bf3275f600e0970b5a4a663890a0eb078`. Source and checkpoint hashes,
matched training streams, software conditions and metrics recomputed from
saved predictions pass the recorded checks. The largest option-order
probability difference is 8.43e-7.

Values below are mean (sample SD) over three training seeds. KL compares the
full distribution with the known posterior; lower is better.

| World | Variant | Posterior KL | Choice accuracy (%) |
|---|---|---:|---:|
| q=0.7 | isolated | 0.1444 (0.0438) | 69.36 (0.87) |
| q=0.7 | count | 0.1848 (0.0677) | 66.99 (2.94) |
| q=0.7 | random context | 0.2028 (0.0440) | 66.74 (1.65) |
| q=0.7 | question pooling | 0.1581 (0.0762) | 68.35 (1.60) |
| odds=7 | isolated | 0.1656 (0.0679) | 70.68 (1.66) |
| odds=7 | count | 0.1105 (0.0198) | 71.45 (0.50) |
| odds=7 | random context | 0.1028 (0.0143) | 71.82 (0.96) |
| odds=7 | question pooling | 0.1190 (0.0284) | 71.82 (0.98) |

The count/pooling interventions do not demonstrate a consistent improvement.
All six paired KL intervals include zero. In the original world, Count has
lower ECE but worse mean KL, TV, NLL and accuracy than Isolated. In the
fixed-odds control, Random context has lower mean KL than Count. These are
limited three-seed, fixed-budget results on the synthetic generator.

The matched lexical controls reach 73.125% accuracy in the original world and
75.650% in the fixed-odds world on the new test set. Their known-noise reference
has the same ranking. The original-world count-conditioned fit has mean KL
about 6.16e-6; the fixed-odds fixed-scale fit has mean KL about 1.20e-6.

The common-environment multitask controls use the legacy test set:

| Training source | Choice accuracy (%) | Array comparison |
|---|---:|---|
| Provided source | 64.475 | Exactly matches the expression control |
| Reviewed source | 50.925 | Exactly matches the earlier fresh checkpoint |
| Reviewed source, original sigmoid expression | 64.475 | Exactly matches provided-source rerun |

All three have identical initial weights and training inputs/targets. The
expression control changes only the mathematically equivalent sigmoid used
for BCE gradients. The 13.55-point difference is a seed-0 numerical sensitivity
finding, not evidence that one expression generally trains better models.
The supplied 66.650% checkpoint is still 2.175 points above the original-source
rerun. The fixed/escape-regime training gaps have not received these controls.

Raw records, all seeds, paired comparisons, secondary temperature scaling,
lexical controls and environment information are in `results/context-study-v1/`.
These Choice-only results use different task mix and test seeds from the older
multitask runs. Raw elapsed durations retain a roughly forty-minute interruption
and are not uninterrupted throughput measurements. The protocol commands
generate full weights and per-request predictions locally.
