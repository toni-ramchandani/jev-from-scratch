"""Replay provided/current stream generators, including binary targets and masks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from context_study import update_data_hash, atomic_json


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = [root/'experiments/reference_numpy/data.py', root/'tiny_s1/data.py']
    modules = [module_at(path, f'replay_source_{i}') for i,path in enumerate(sources)]
    generators = [m.make_rng(1000) for m in modules]
    observed = [hashlib.sha256(), hashlib.sha256()]
    binary = [hashlib.sha256(), hashlib.sha256()]
    fields = ('ids', 'pos', 'seg', 'opt_rows', 'opt_group', 'opt_target',
              'noul_rows', 'noul_target', 'mask_bias')
    for _ in range(args.steps):
        batches = [m.collate([m.sample_request(rng) for _ in range(args.batch)])
                   for m,rng in zip(modules, generators)]
        for field in fields:
            assert batches[0][field].dtype == batches[1][field].dtype
            assert np.array_equal(batches[0][field], batches[1][field]), field
        for digest, target_digest, batch in zip(observed, binary, batches):
            update_data_hash(digest, batch)
            target_digest.update(batch['noul_target'].tobytes())
    result = {'role': 'Independent deterministic data replay, not a training execution',
              'steps': args.steps, 'batch': args.batch, 'seed': 1000, 'numpy': np.__version__,
              'fields_compared_exactly': list(fields), 'all_arrays_identical': True,
              'source_sha256': {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'execution_fingerprint': [h.hexdigest() for h in observed],
              'binary_target_sha256': [h.hexdigest() for h in binary]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
