"""Verify matched executions and aggregate every registered study condition."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from context_study import atomic_json, score_metrics

METRICS = ('posterior_kl', 'tv_to_bayes', 'nll', 'accuracy', 'brier', 'ece')
COMPARISONS = (('count', 'isolated'), ('count', 'random_count'), ('set', 'isolated'))


def aggregate(values):
    values = np.asarray(values, float)
    return {'n_seeds': len(values), 'values': values.tolist(), 'mean': float(values.mean()),
            'sd': float(values.std(ddof=1)) if len(values) > 1 else None}


def paired(values):
    out = aggregate(values)
    if len(values) == 3:
        radius = 4.302652729911275 * out['sd'] / np.sqrt(3.)
        out['t_interval_95'] = [out['mean'] - radius, out['mean'] + radius]
    return out


def read_predictions(path, prefix):
    with np.load(path, allow_pickle=False) as arrays:
        return {key[len(prefix)+1:]: arrays[key] for key in arrays.files if key.startswith(prefix+'_')}


def context_association(counts):
    counts = np.asarray(counts, float)[2:7, 2:7]
    expected = counts.sum(axis=1, keepdims=True) @ counts.sum(axis=0, keepdims=True) / counts.sum()
    chi2 = ((counts - expected)**2 / expected).sum()
    return float(np.sqrt(chi2 / (counts.sum() * 4.)))


def weight_difference(first, second):
    per_array = []
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in sorted(a.files, key=lambda name:int(name.split('_')[1])):
            delta = a[key].astype(float) - b[key].astype(float)
            per_array.append({'array': key, 'max_abs': float(np.abs(delta).max()),
                              'squared_l2': float((delta**2).sum()),
                              'identical': bool(np.array_equal(a[key], b[key]))})
    return {'max_abs': max(row['max_abs'] for row in per_array),
            'l2': float(np.sqrt(sum(row['squared_l2'] for row in per_array))), 'arrays': per_array}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.study_dir.resolve()
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    progress = json.loads((root/'progress.json').read_text(encoding='utf-8'))
    if progress['counts']['pending'] or progress['counts']['running']:
        raise RuntimeError('Wait for every registered execution before final aggregation')
    rows, missing = {}, []
    fingerprints = {}
    for world, seed, mode in itertools.product(protocol['worlds'], protocol['seeds'], protocol['modes']):
        folder = root/'runs'/world/mode/f'seed-{seed}'
        path = folder/'results.json'
        key = (world, mode, seed)
        if not path.exists():
            missing.append({'world': world, 'mode': mode, 'seed': seed, 'status': 'failed'})
            continue
        result = json.loads(path.read_text(encoding='utf-8'))
        config = result['config']
        assert all(config[k] == v for k, v in zip(('world', 'mode', 'seed'), key))
        assert config['steps'] == protocol['steps'] and config['batch'] == protocol['batch']
        assert not result['pilot'] and config['parameters'] == 116993
        assert config['context_parameters'] == 448
        assert result['option_order_max_abs_difference'] <= 1e-4
        assert hashlib.sha256((folder/'model.npz').read_bytes()).hexdigest() == result['checkpoint_sha256']
        for split, prefix in (('test', 'test'), ('heldout_vocabulary', 'heldout')):
            predictions = read_predictions(folder/'predictions.npz', prefix)
            recomputed = score_metrics(predictions)
            for metric in METRICS:
                np.testing.assert_allclose(recomputed[metric], result[split][metric], atol=1e-12, rtol=1e-12)
            # Verify identical test outcomes and true posteriors across treatments.
            digest = hashlib.sha256()
            for name in ('n', 'targets', 'bayes'):
                digest.update(predictions[name].tobytes())
            fingerprints.setdefault((world, split), set()).add(digest.hexdigest())
        if mode == 'random_count':
            result['random_context_cramers_v'] = context_association(result['option_count_by_context_index'])
        rows[key] = result
    assert all(len(values) == 1 for values in fingerprints.values())
    blocks = []
    lexical = json.loads((root/'lexical-controls.json').read_text(encoding='utf-8'))
    for world, seed in itertools.product(protocol['worlds'], protocol['seeds']):
        available = [rows[(world, mode, seed)] for mode in protocol['modes'] if (world, mode, seed) in rows]
        if not available:
            continue
        hashes = {r['training_data_sha256'] for r in available}
        initial = {r['config']['initial_parameters_sha256'] for r in available}
        assert len(hashes) == len(initial) == 1
        lexical_rows = [r for r in lexical['runs'] if r['world'] == world and r['seed'] == seed]
        assert len(lexical_rows) == 2 and {r['training_data_sha256'] for r in lexical_rows} == hashes
        blocks.append({'world': world, 'seed': seed, 'models_verified': len(available),
                       'training_data_sha256': next(iter(hashes)),
                       'initial_parameters_sha256': next(iter(initial)), 'lexical_data_match': True})
    source_commits = {r['config']['source_commit'] for r in rows.values()}
    source_hashes = {json.dumps(r['config']['source_sha256'], sort_keys=True) for r in rows.values()}
    assert len(source_commits) == len(source_hashes) == 1 and None not in source_commits
    reproduction, reproduction_missing = {}, []
    for variant in protocol['reproduction_controls']:
        path = root/'reproduction'/variant/'results.json'
        if path.exists():
            reproduction[variant] = json.loads(path.read_text(encoding='utf-8'))
        else:
            reproduction_missing.append({'variant': variant, 'status': 'failed'})
    if reproduction:
        assert len({r['execution']['training_data_sha256'] for r in reproduction.values()}) == 1
        assert len({r['execution']['initial_parameter_sha256'] for r in reproduction.values()}) == 1
        assert all(r['config']['source_commit'] in source_commits for r in reproduction.values())
    reproduction_differences = {}
    for first, second in itertools.combinations(reproduction, 2):
        reproduction_differences[first+' minus '+second] = weight_difference(
            root/'reproduction'/first/'checkpoints/ckpt_stream.npz',
            root/'reproduction'/second/'checkpoints/ckpt_stream.npz')
    groups, comparisons, interaction = [], [], []
    for world, mode in itertools.product(protocol['worlds'], protocol['modes']):
        selected = [rows[(world, mode, seed)] for seed in protocol['seeds'] if (world, mode, seed) in rows]
        if selected:
            groups.append({'world': world, 'mode': mode,
                           **{split: {metric: aggregate([r[split][metric] for r in selected]) for metric in METRICS}
                              for split in ('test', 'heldout_vocabulary')},
                           'temperature': aggregate([r['temperature'] for r in selected]),
                           'test_temperature_scaled': {metric: aggregate([r['test_temperature_scaled'][metric] for r in selected])
                                                       for metric in METRICS}})
    for world in protocol['worlds']:
        for first, second in COMPARISONS:
            seeds = [s for s in protocol['seeds'] if (world, first, s) in rows and (world, second, s) in rows]
            if seeds:
                comparisons.append({'world': world, 'comparison': first+' minus '+second, 'seeds': seeds,
                                    'test': {m: paired([rows[(world, first, s)]['test'][m] - rows[(world, second, s)]['test'][m]
                                                        for s in seeds]) for m in METRICS}})
    for first, second in COMPARISONS:
        seeds = [s for s in protocol['seeds'] if all((w, mode, s) in rows for w in protocol['worlds'] for mode in (first, second))]
        if seeds:
            interaction.append({'comparison': first+' minus '+second,
                                'world_difference': 'fixed_q minus fixed_odds', 'role': 'Exploratory',
                                'test': {m: paired([(rows[('fixed_q', first, s)]['test'][m] - rows[('fixed_q', second, s)]['test'][m])
                                                    - (rows[('fixed_odds', first, s)]['test'][m] - rows[('fixed_odds', second, s)]['test'][m])
                                                   for s in seeds]) for m in protocol['primary_metrics']}})
    summary = {'study_id': protocol['study_id'], 'source_commit': next(iter(source_commits)),
               'protocol_sha256': progress['protocol_sha256'], 'counts': progress['counts'],
               'failed_context_runs': missing, 'failed_reproduction_runs': reproduction_missing,
               'integrity': {'matched_blocks': blocks, 'shared_source_sha256': json.loads(next(iter(source_hashes))),
                             'prediction_metrics_recomputed': True, 'test_targets_and_posteriors_match': True,
                             'max_option_order_error': max(r['option_order_max_abs_difference'] for r in rows.values())},
               'groups': groups, 'paired_comparisons': comparisons, 'exploratory_interactions': interaction,
               'uncertainty': 'SD across training seeds; paired t intervals with df=2 when all three seeds complete. No multiplicity-adjusted significance claims.',
               'reproduction': reproduction, 'reproduction_weight_differences': reproduction_differences,
               'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(root/'summary.json', summary)
    atomic_json(root/'all-context-runs.json', [{'world':w, 'mode':m, 'seed':s, **r} for (w,m,s),r in rows.items()])
    with (root/'summary.csv').open('w', encoding='utf-8', newline='') as handle:
        columns = ['world', 'mode', 'seed', 'split'] + list(METRICS)
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for (world, mode, seed), result in rows.items():
            for split in ('test', 'heldout_vocabulary', 'test_temperature_scaled'):
                writer.writerow({'world': world, 'mode': mode, 'seed': seed, 'split': split,
                                 **{m:result[split][m] for m in METRICS}})
    print(json.dumps({'groups': len(groups), 'context_runs': len(rows), 'reproduction_runs': len(reproduction),
                      'failed_runs': len(missing) + len(reproduction_missing), 'integrity': 'passed'}))


if __name__ == '__main__':
    main()
