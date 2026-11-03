"""Execute a fixed option-context protocol with a bounded local worker pool."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from context_study import atomic_json
from reproduce_training import source_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    jobs = [{'kind': 'context', 'world': world, 'mode': mode, 'seed': seed, 'status': 'pending'}
            for world in protocol['worlds'] for seed in protocol['seeds'] for mode in protocol['modes']]
    random.Random(protocol['job_order_seed']).shuffle(jobs)
    jobs = [{'kind':'reproduction','variant':variant,'seed':0,'status':'pending'}
            for variant in protocol.get('reproduction_controls', [])] + jobs
    if args.dry_run:
        print(json.dumps({'jobs': len(jobs), 'protocol_sha256': digest, 'order': jobs}, indent=2))
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        old = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
        if old != protocol:
            raise ValueError('Existing directory belongs to a different protocol')
    atomic_json(output / 'protocol.json', protocol)
    source = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                     for name in ('context_study.py', 'data.py', 'autograd.py', 'model.py', 'train.py', 'baselines.py')}
    env = dict(os.environ)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    active = {}
    started = time.time()
    last_print = 0.
    for job in jobs:
        run_dir = (output/'runs'/job['world']/job['mode']/f"seed-{job['seed']}" if job['kind']=='context'
                   else output/'reproduction'/job['variant'])
        job['directory'] = str(run_dir)
        result = run_dir/'results.json'
        if result.exists():
            saved = json.loads(result.read_text(encoding='utf-8'))
            config = saved['config']
            if job['kind']=='context':
                valid = (not saved.get('pilot') and 'test' in saved
                         and all(config[k] == job[k] for k in ('world', 'mode', 'seed'))
                         and config['steps'] == protocol['steps'] and config['batch'] == protocol['batch']
                         and config['source_sha256'] == source_hashes)
            else:
                reference = {name:hashlib.sha256((source_directory(job['variant'])/name).read_bytes()).hexdigest()
                             for name in ('autograd.py','data.py','model.py','train.py')}
                valid = (config['variant']==job['variant'] and config['steps']==protocol['steps']
                         and config['batch']==protocol['batch'] and config['source_sha256']==reference
                         and 'choice' in saved)
            if not valid:
                raise ValueError(f'Existing result does not match this execution: {run_dir}')
            job['status'] = 'complete'
    try:
        while any(j['status'] in ('pending', 'running') for j in jobs):
            for i in list(active):
                process, log = active[i]
                code = process.poll()
                if code is None:
                    continue
                log.close()
                job = jobs[i]
                job['exit_code'] = code
                if code == 0 and (Path(job['directory'])/'results.json').exists():
                    saved = json.loads((Path(job['directory'])/'results.json').read_text(encoding='utf-8'))
                    valid = ('test' in saved and not saved.get('pilot')) if job['kind']=='context' else 'choice' in saved
                    job['status'] = 'complete' if valid else 'failed'
                else:
                    job['status'] = 'failed'
                job['ended_unix'] = time.time()
                del active[i]
            for i, job in enumerate(jobs):
                if len(active) >= args.workers:
                    break
                if job['status'] != 'pending':
                    continue
                run_dir = Path(job['directory'])
                run_dir.mkdir(parents=True, exist_ok=True)
                log = (run_dir/'run.log').open('a', encoding='utf-8')
                if job['kind']=='context':
                    command = [sys.executable, str(source/'context_study.py'), '--world', job['world'],
                               '--mode', job['mode'], '--seed', str(job['seed']), '--steps', str(protocol['steps']),
                               '--batch', str(protocol['batch']), '--output-dir', str(run_dir)]
                else:
                    command = [sys.executable, str(source/'reproduce_training.py'), '--variant', job['variant'],
                               '--steps', str(protocol['steps']), '--batch', str(protocol['batch']),
                               '--output-dir', str(run_dir)]
                process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                job.update(status='running', pid=process.pid, started_unix=time.time())
                active[i] = (process, log)
            counts = {status: sum(j['status'] == status for j in jobs)
                      for status in ('pending', 'running', 'complete', 'failed')}
            state = {'protocol_sha256': digest, 'driver_pid': os.getpid(),
                     'elapsed_seconds': time.time()-started, 'counts': counts, 'jobs': jobs}
            atomic_json(output/'progress.json', state)
            if time.time()-last_print >= 30:
                print(json.dumps({'elapsed_seconds': round(state['elapsed_seconds']), **counts}), flush=True)
                last_print = time.time()
            if active:
                time.sleep(2)
        print(json.dumps({'finished': True, **counts}), flush=True)
        if counts['failed']:
            raise SystemExit(1)
    except KeyboardInterrupt:
        for process, log in active.values():
            process.terminate()
            log.close()
        raise


if __name__ == '__main__':
    main()
