"""Re-execute provided/current multitask code under a common local environment."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from context_study import atomic_json
from evaluate import load, make_set, run_model, option_metrics, noul_metrics


def source_directory(variant):
    here = Path(__file__).resolve().parent
    return here.parent/'experiments/reference_numpy' if variant == 'provided_code' else here


def weight_hash(path):
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as arrays:
        for key in sorted(arrays.files, key=lambda x:int(x.split('_')[1])):
            value = arrays[key]
            digest.update((str(value.shape)+str(value.dtype)).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['provided_code','current_code','current_original_sigmoid'], required=True)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        raise FileExistsError('Completed result exists')
    snapshot = output/'source'
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for name in ('autograd.py','data.py','model.py','train.py'):
        source = (source_directory(args.variant)/name).read_bytes()
        (snapshot/name).write_bytes(source)
        hashes[name] = hashlib.sha256(source).hexdigest()
    checkpoints = output/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    tracking = output/'execution.json'
    launcher = '''
import sys,hashlib,json,runpy,os
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1])
import data as D
from model import TinyS1
if os.environ.get('REPRO_ORIGINAL_SIGMOID') == '1':
    import autograd as ag
    ag.sigmoid_np=lambda z:1.0/(1.0+np.exp(-z))
tracking=Path(sys.argv[2]); train_path=sys.argv[3]; arguments=sys.argv[4:]
digest=hashlib.sha256(); original=D.collate; constructor=TinyS1.__init__; initial=[]
def capture_init(self,*args,**kwargs):
    constructor(self,*args,**kwargs)
    h=hashlib.sha256()
    for p in self.parameters():
        h.update(str(p.shape).encode());h.update(p.data.tobytes())
    initial.append(h.hexdigest())
TinyS1.__init__=capture_init
def capture_batch(requests):
    batch=original(requests)
    for name in ('ids','pos','seg','opt_rows','opt_group','opt_target'):
        a=np.ascontiguousarray(batch[name]);digest.update((name+str(a.shape)+str(a.dtype)).encode());digest.update(a.tobytes())
    return batch
D.collate=capture_batch
sys.argv=[train_path]+arguments
runpy.run_path(train_path,run_name='__main__')
tracking.write_text(json.dumps({'training_data_sha256':digest.hexdigest(),'initial_parameter_sha256':initial[0],
                              'numpy':np.__version__,'python':sys.version}))
'''
    command = [sys.executable, '-c', launcher, str(snapshot), str(tracking), str(snapshot/'train.py'),
               '--run', 'stream', '--steps', str(args.steps), '--batch', str(args.batch), '--seed', '0']
    if args.variant != 'provided_code':
        command += ['--output-dir', str(checkpoints)]
    start = time.perf_counter()
    env = os.environ.copy()
    env['REPRO_ORIGINAL_SIGMOID'] = '1' if args.variant == 'current_original_sigmoid' else '0'
    subprocess.run(command, cwd=checkpoints, env=env, check=True,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    training_seconds = time.perf_counter()-start
    records = run_model(load('stream', checkpoints), make_set(5002, 4000))
    old = Path(__file__).resolve().parents[1]/'artifacts'
    result = {'config': {'variant':args.variant, 'steps':args.steps,'batch':args.batch,'seed':0,
                         'source_commit':os.environ.get('EXPERIMENT_SOURCE_COMMIT'),
                         'source_sha256':hashes,'test_seed':5002,'test_requests':4000},
              'execution':json.loads(tracking.read_text()), 'training_seconds':training_seconds,
              'choice':option_metrics(records,'choice'), 'score':option_metrics(records,'score'),
              'noul':noul_metrics(records), 'weight_data_sha256':weight_hash(checkpoints/'ckpt_stream.npz'),
              'reference_weight_data_sha256':{name:weight_hash(old/name/'ckpt_stream.npz') for name in ('supplied','fresh')}}
    result['identical_to_reference'] = {name: value == result['weight_data_sha256']
                                       for name,value in result['reference_weight_data_sha256'].items()}
    result['total_seconds'] = time.perf_counter()-start
    atomic_json(output/'results.json', result)
    print(json.dumps({'variant':args.variant,'choice_accuracy':result['choice']['acc'],
                      'identical_to_reference':result['identical_to_reference']}), flush=True)


if __name__ == '__main__':
    main()
