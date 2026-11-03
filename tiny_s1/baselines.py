"""Lexical Choice baselines using only state words and option descriptions.

Counts exact word matches, then fits one shared logit scale or a two-parameter
scale conditioned on log(option_count - 1). No Bayes probabilities are used in
fitting. The known-noise reference uses the generator's q and is labelled as an
oracle. This evaluates Choice only, not Score, Noul or real-language capability.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import data as D
from evaluate import make_set, reliability, wilson_interval


def features(requests):
    n = np.array([len(r["questions"][0]["options"]) for r in requests])
    width = int(n.max())
    counts = np.zeros((len(requests), width))
    bayes = np.zeros_like(counts)
    targets = np.zeros(len(requests), dtype=int)
    for i, request in enumerate(requests):
        q = request["questions"][0]
        targets[i] = int(np.argmax(q["target"]))
        for j, (_, description) in enumerate(q["options"]):
            words = set(description)
            counts[i, j] = sum(word in words for word in request["state"])
        bayes[i, :n[i]] = q["bayes"]
    return counts, n, targets, bayes


def design(counts, n, with_count):
    return np.stack([counts, counts * np.log(n - 1)[:, None]], axis=-1) if with_count else counts[..., None]


def distribution(beta, x, n):
    scores = x @ beta
    valid = np.arange(scores.shape[1])[None, :] < n[:, None]
    scores = np.where(valid, scores, -np.inf)
    maximum = scores.max(axis=1, keepdims=True)
    exponent = np.exp(scores - maximum)
    p = exponent / exponent.sum(axis=1, keepdims=True)
    log_p = scores - maximum - np.log(exponent.sum(axis=1, keepdims=True))
    return p, log_p


def fit(counts, n, y, with_count):
    x = design(counts, n, with_count)
    beta = np.ones(x.shape[-1])
    index = np.arange(len(y))
    for iteration in range(40):
        p, log_p = distribution(beta, x, n)
        loss = float(-log_p[index, y].mean())
        mean_x = np.einsum('nc,ncd->nd', p, x)
        gradient = (mean_x - x[index, y]).mean(axis=0)
        if np.max(np.abs(gradient)) < 1e-9:
            break
        second = np.einsum('nc,ncd,nce->de', p, x, x) / len(y)
        hessian = second - mean_x.T @ mean_x / len(y)
        step = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-12, gradient)
        rate = 1.0
        for _ in range(30):
            proposal = beta - rate * step
            _, candidate = distribution(proposal, x, n)
            new_loss = float(-candidate[index, y].mean())
            if new_loss <= loss - 1e-4 * rate * float(gradient @ step):
                beta = proposal
                break
            rate *= .5
        else:
            break
    return beta, {'iterations': iteration + 1, 'gradient_max_abs': float(np.abs(gradient).max()),
                  'training_nll': float(-distribution(beta, x, n)[1][index, y].mean())}


def metrics(p, log_p, y, bayes):
    index = np.arange(len(y))
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    brier = (p * p).sum(axis=1) - 2 * p[index, y] + 1
    ece, bins = reliability(p.max(axis=1), correct)
    return {'n': len(y), 'accuracy': float(correct.mean()), 'accuracy_wilson_95': wilson_interval(correct),
            'nll': float(-log_p[index, y].mean()), 'brier': float(brier.mean()), 'ece': ece,
            'tv_to_bayes': float((.5 * np.abs(p - bayes).sum(axis=1)).mean()), 'reliability': bins}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-requests', type=int, default=128000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--output', type=Path, default=Path('../results/lexical-baselines.json'))
    args = ap.parse_args()
    if args.train_requests < 1:
        ap.error('train-requests must be positive')
    start = time.perf_counter()
    sets = {'test': features(make_set(5002, 4000)),
            'heldout_vocabulary': features(make_set(5008, 2000, topics=D.HELD_OUT, n_opts=(2, 4)))}
    result = {'config': {'train_requests_per_seed': args.train_requests, 'seeds': args.seeds,
                          'test_seed': 5002, 'heldout_seed': 5008, 'numpy': np.__version__,
                          'input_features': 'Exact state-word membership in each supplied option description',
                          'scope': 'Choice only; no comparison of Score, Noul or real-language capability'},
              'runs': [], 'known_noise_oracle': {}}
    for name, (counts, n, y, bayes) in sets.items():
        beta = np.array([np.log(D.Q_TOPIC / (1 - D.Q_TOPIC)), 1.0])
        p, log_p = distribution(beta, design(counts, n, True), n)
        np.testing.assert_allclose(p, bayes, atol=1e-14, rtol=1e-14)
        result['known_noise_oracle'][name] = metrics(p, log_p, y, bayes)
    for seed in args.seeds:
        train = features(make_set(1000 + seed, args.train_requests))
        counts, n, y, _ = train
        fingerprint = hashlib.sha256(counts.tobytes() + n.tobytes() + y.tobytes()).hexdigest()
        for with_count in (False, True):
            beta, status = fit(counts, n, y, with_count)
            row = {'seed': seed, 'model': 'count_conditioned' if with_count else 'fixed_scale',
                   'parameters': beta.tolist(), 'fit': status, 'training_feature_sha256': fingerprint,
                   'evaluations': {}}
            for name, (test_counts, test_n, test_y, test_bayes) in sets.items():
                p, log_p = distribution(beta, design(test_counts, test_n, with_count), test_n)
                row['evaluations'][name] = metrics(p, log_p, test_y, test_bayes)
            result['runs'].append(row)
            print(json.dumps({'seed': seed, 'model': row['model'], 'parameters': beta.tolist(),
                              'test_accuracy': row['evaluations']['test']['accuracy'],
                              'test_nll': row['evaluations']['test']['nll'] }), flush=True)
    result['seconds'] = time.perf_counter() - start
    result['limitations'] = [
        'Hand-designed lexical features suit the constructed cue-word generator.',
        'No new transformer training seeds were run in this baseline experiment.',
        'This is not a causal ablation of the transformer mask or its optimization.',
        'Fixed-scale and count-conditioned models have the same ranking for positive scales.',
        'Option count changes the true probability scale, not the Bayes-optimal cue-count ranking.'
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
