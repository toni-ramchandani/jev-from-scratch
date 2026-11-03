"""Supplementary lexical controls on the matched option-context training streams."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import data as D
from baselines import features, design, fit
from context_study import (choice_request, make_requests, update_data_hash,
                           score_metrics, atomic_json)


def records(counts, n, y, bayes, beta, with_count):
    scores = design(counts, n, with_count) @ beta
    scores = np.where(np.arange(scores.shape[1])[None] < n[:, None], scores, -np.inf)
    return {'scores': scores, 'n': n, 'targets': y, 'bayes': bayes}


def training_features(world, seed, steps, batch_size):
    rng, digest = D.make_rng(1000 + seed), hashlib.sha256()
    columns = [[], [], [], []]
    for _ in range(steps):
        requests = [choice_request(rng, world) for _ in range(batch_size)]
        batch = D.collate(requests)
        update_data_hash(digest, batch)
        values = features(requests)
        for index, value in enumerate(values):
            if index in (0, 3) and value.shape[1] < 6:
                value = np.pad(value, ((0, 0), (0, 6 - value.shape[1])))
            columns[index].append(value)
    return tuple(np.concatenate(column) for column in columns), digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Completed controls already exist')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    start = time.perf_counter()
    result = {'config': {
        'study_id': protocol['study_id'], 'role': 'Supplementary, handcrafted lexical controls',
        'steps': protocol['steps'], 'batch': protocol['batch'], 'seeds': protocol['seeds'],
        'fit': 'Maximum likelihood on sampled outcomes; Bayes probabilities excluded from fitting',
        'source_commit': os.environ.get('EXPERIMENT_SOURCE_COMMIT'),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        'numpy': np.__version__}, 'runs': [], 'known_noise_oracle': {}}
    for world in protocol['worlds']:
        sets = {
            'test': features(make_requests(world, protocol['test']['seed'], protocol['test']['requests'])),
            'heldout_vocabulary': features(make_requests(world, protocol['heldout_vocabulary']['seed'],
                                                       protocol['heldout_vocabulary']['requests'], heldout=True))}
        oracle_beta = np.array([np.log(.7/.3), 1.]) if world == 'fixed_q' else np.array([np.log(7.), 0.])
        result['known_noise_oracle'][world] = {}
        for split, values in sets.items():
            oracle = records(*values, oracle_beta, True)
            score = oracle['scores']
            exp = np.exp(score - score.max(axis=1, keepdims=True))
            np.testing.assert_allclose(exp / exp.sum(axis=1, keepdims=True), values[3], atol=1e-14, rtol=1e-14)
            result['known_noise_oracle'][world][split] = score_metrics(oracle)
        for seed in protocol['seeds']:
            train, data_hash = training_features(world, seed, protocol['steps'], protocol['batch'])
            counts, n, y, _ = train
            for with_count in (False, True):
                beta, diagnostics = fit(counts, n, y, with_count)
                if diagnostics['gradient_max_abs'] > 1e-7:
                    raise ArithmeticError('Lexical fit failed stationarity check')
                row = {'world': world, 'seed': seed,
                       'model': 'count_conditioned' if with_count else 'fixed_scale',
                       'parameters': beta.tolist(), 'fit': diagnostics,
                       'training_data_sha256': data_hash,
                       'evaluations': {split: score_metrics(records(*values, beta, with_count))
                                       for split, values in sets.items()}}
                result['runs'].append(row)
                print(json.dumps({'world': world, 'seed': seed, 'model': row['model'],
                                  'posterior_kl': row['evaluations']['test']['posterior_kl']}), flush=True)
    result['seconds'] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({'finished': True, 'fits': len(result['runs']), 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
