"""Frozen eight-run digits protocol. Run `freeze` after preflight, then `run`."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

# Source lives under <repo>/tiny_s1/numerical/digits. Generated data and
# outputs default to a sibling of <repo>, never into the repository.
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import mlp

HERE=Path(__file__).resolve().parent
DEFAULT_OUT=HERE.parents[2].parent/'digits-numerical-output'
DEFAULT_DATA=DEFAULT_OUT/'data'
RUN_LIST=[dict(mode='shared',seed=s,expression=e) for s in mlp.CONFIG['seeds']
          for e in ('stable','direct')]+[
          dict(mode='detached',seed=0,expression=e) for e in ('stable','direct')]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    result={name:file_hash(HERE/name) for name in ('mlp.py','train.py','validate.py','fetch_data.py')}
    assert json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))['training']==mlp.CONFIG
    result['protocol']=file_hash(HERE/'protocol.json')
    return result


def data_hashes(folder):
    return {name:file_hash(folder/name) for name in ('optdigits.tra','optdigits.tes','optdigits.names')}


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def gate_check(path):
    gate=json.loads(path.read_text(encoding='utf-8'))
    assert gate.get('all_pass') is True, 'preflight is not passing'
    assert gate['source_hashes']==source_hashes(), 'source/protocol changed since preflight; rerun validator'
    return gate


def freeze(args):
    gate=gate_check(args.gate)
    dh=data_hashes(args.data_dir)
    expected=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))['dataset']['files']
    assert all(dh[n]==entry['sha256'] for n,entry in expected.items()), 'Data differs from official pinned bytes'
    assert dh['optdigits.tra']==gate['training_file_sha256']
    assert not (args.out_dir/'freeze.json').exists(), 'freeze already exists; do not overwrite'
    assert not (args.out_dir/'runs').exists(), 'runs already exist; do not retroactively freeze'
    args.out_dir.mkdir(parents=True,exist_ok=True)
    frozen=args.out_dir/'source';frozen.mkdir()
    for name in ('mlp.py','train.py','validate.py','fetch_data.py'):
        shutil.copy2(HERE/name,frozen/name)
    shutil.copy2(HERE/'protocol.json',frozen/'protocol.json')
    shutil.copy2(HERE/'protocol.json',args.out_dir/'protocol.json')
    if args.gate.resolve() != (args.out_dir/'preflight.json').resolve():
        shutil.copy2(args.gate,args.out_dir/'preflight.json')
    manifest=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_hashes=source_hashes(), data_sha256=dh, config=mlp.CONFIG,
                  run_list=RUN_LIST, preflight_sha256=file_hash(args.gate),
                  python=platform.python_version(), numpy=np.__version__,
                  platform=platform.platform(), processor=platform.processor(),
                  thread_settings={k:os.environ[k] for k in
                      ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                  blas_configuration=str(getattr(np.__config__,'CONFIG','unavailable')),
                  dataset=dict(doi='10.24432/C50P49',license='CC BY 4.0',
                      source='https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits',
                      changes='Original train/test split preserved; pixels divided by16; odd parity derived'))
    write_json(args.out_dir/'freeze.json',manifest)
    print(json.dumps(dict(status='frozen',manifest=str(args.out_dir/'freeze.json'),runs=8)))


def max_difference(a,b,names=mlp.NAMES):
    return max(float(np.max(np.abs(a[k]-b[k]))) for k in names)


def run_pair(seed,detached,train_data,test_data,out):
    x,y,parity=train_data
    tx,ty,_=test_data
    mode='detached' if detached else 'shared'
    pair=out/'runs'/mode/f'seed-{seed}'
    assert not pair.exists(), f'Run directory exists: {pair}'
    pair.mkdir(parents=True)
    initial=mlp.initialize(seed);order=mlp.permutations(seed)
    initial_hash=mlp.hash_arrays(initial)
    permutation_hash=mlp.hash_arrays(dict(permutations=order))
    np.savez_compressed(pair/'initialization.npz',**initial)
    np.savez_compressed(pair/'permutations.npz',permutations=order)
    params=[mlp.clone(initial),mlp.clone(initial)]
    states=[mlp.new_optimizer(p) for p in params]
    expressions=('stable','direct')
    probe_x=x[:128]
    history=[];anchors=[]
    first_gradient=None;first_parameter=None;first_backbone=None
    overflow_counts=[0,0]
    started=time.time()
    metadata=dict(mode=mode,seed=seed,init_sha256=initial_hash,
                  permutations_sha256=permutation_hash,config=mlp.CONFIG,
                  training_arrays_sha256=mlp.hash_arrays(dict(x=x,y=y,parity=parity)),
                  test_arrays_sha256=mlp.hash_arrays(dict(x=tx,y=ty)),
                  training_rows=3823,test_rows=1797,updates=1500)
    write_json(pair/'config.json',metadata)

    def anchor(update):
        predictions=[]
        for expression,p,state in zip(expressions,params,states):
            folder=pair/expression;folder.mkdir(exist_ok=True)
            mlp.checkpoint(folder/f'checkpoint-{update:04d}.npz',p,state)
            pred=mlp.evaluation_predictions(p,probe_x)
            np.savez_compressed(folder/f'training-probe-{update:04d}.npz',**pred)
            predictions.append(pred)
        anchors.append(dict(update=update,role='Fixed first128 training rows; no held-out evaluation',
                            prediction_distance=mlp.distances(*predictions),
                            max_parameter_difference=max_difference(*params),
                            max_backbone_parameter_difference=max_difference(*params,mlp.NAMES[:4])))

    anchor(0)
    for update,(epoch,rows) in enumerate(mlp.batches(order),1):
        gradients=[];losses=[];ranges=[]
        for j,(expression,p,state) in enumerate(zip(expressions,params,states)):
            loss,g,c=mlp.loss_and_grad(p,x[rows],y[rows],parity[rows],expression,detached)
            gradients.append(g)
            losses.append(dict(total=float(loss),digit=float(c['digit_loss']),parity=float(c['parity_loss'])))
            z=c['parity']
            with np.errstate(over='ignore'):
                count=int(np.sum(~np.isfinite(np.exp(-z))))
            overflow_counts[j]+=count
            ranges.append(dict(min=float(z.min()),max=float(z.max()),
                               direct_exp_overflow_inputs=count,nonfinite_logits=0))
        gd=max_difference(*gradients)
        if first_gradient is None and gd>0:first_gradient=update
        for p,g,state in zip(params,gradients,states):mlp.adam_step(p,g,state)
        pd=max_difference(*params);bd=max_difference(*params,mlp.NAMES[:4])
        if first_parameter is None and pd>0:first_parameter=update
        if first_backbone is None and bd>0:first_backbone=update
        history.append(dict(update=update,epoch=epoch+1,batch_size=len(rows),
                            max_gradient_difference=gd,max_parameter_difference=pd,
                            max_backbone_parameter_difference=bd,
                            losses=dict(zip(expressions,losses)),binary_logit_ranges=dict(zip(expressions,ranges))))
        if update in mlp.CONFIG['probe_updates']:anchor(update)
        if update%25==0 or update==1500:
            write_json(pair/'progress.json',dict(update=update,total_updates=1500,
                       elapsed_seconds=time.time()-started,first_gradient_difference_update=first_gradient,
                       first_parameter_difference_update=first_parameter,
                       max_parameter_difference=pd))
    assert update==1500 and len(rows)==111
    final=[];scores={}
    for expression,p,state in zip(expressions,params,states):
        pred=mlp.evaluation_predictions(p,tx)
        folder=pair/expression
        np.savez_compressed(folder/'test-predictions-final.npz',**pred,y=ty,parity=ty%2)
        score=mlp.quality(pred,ty);scores[expression]=score
        write_json(folder/'metrics-final.json',score)
        final.append(pred)
    distance=mlp.distances(*final)
    if detached:
        digit_names=('w1','b1','w2','b2','wd','bd')
        assert max_difference(*params,digit_names)==0, 'Detached control digit arrays differ'
        assert distance['digit_mean_tv']==0 and distance['digit_argmax_disagreements']==0
    result=dict(mode=mode,seed=seed,complete=True,elapsed_seconds=time.time()-started,
                first_gradient_difference_update=first_gradient,
                first_parameter_difference_update=first_parameter,
                first_backbone_difference_update=first_backbone,
                direct_exp_overflow_input_counts=dict(zip(expressions,overflow_counts)),
                proper_score_and_accuracy=scores,
                direct_minus_stable={k:scores['direct'][k]-scores['stable'][k] for k in scores['stable']},
                final_test_prediction_distance=distance,
                dtype_checks='All parameters, gradients, forward/backward floating intermediates and Adam states FP32',
                fixed_training_probe_anchors=anchors)
    write_json(pair/'trajectory.json',history)
    write_json(pair/'results.json',result)
    return result,dict(zip(expressions,final))


def run(args):
    gate=gate_check(args.gate)
    frozen=json.loads((args.out_dir/'freeze.json').read_text(encoding='utf-8'))
    assert frozen['source_hashes']==source_hashes()
    assert frozen['data_sha256']==data_hashes(args.data_dir)
    assert frozen['preflight_sha256']==file_hash(args.gate)
    assert frozen['config']==mlp.CONFIG and frozen['run_list']==RUN_LIST
    train_data=mlp.load_split(args.data_dir/'optdigits.tra','tra')
    test_data=mlp.load_split(args.data_dir/'optdigits.tes','tes')
    assert gate['training_array_sha256']==mlp.hash_arrays(dict(zip(('x','y','parity'),train_data)))
    results=[];predictions={}
    for seed in mlp.CONFIG['seeds']:
        result,pred=run_pair(seed,False,train_data,test_data,args.out_dir)
        results.append(result);predictions[seed]=pred
    negative,_=run_pair(0,True,train_data,test_data,args.out_dir)
    between={expression:[dict(seed_a=a,seed_b=b,
                 distance=mlp.distances(predictions[a][expression],predictions[b][expression]))
                for a,b in itertools.combinations(mlp.CONFIG['seeds'],2)]
             for expression in ('stable','direct')}
    summary={k:dict(mean=float(np.mean([r['direct_minus_stable'][k] for r in results])),
                   min=min(r['direct_minus_stable'][k] for r in results),
                   max=max(r['direct_minus_stable'][k] for r in results))
             for k in results[0]['direct_minus_stable']}
    within_mean=float(np.mean([r['final_test_prediction_distance']['digit_mean_tv'] for r in results]))
    baseline=float(np.mean([r['distance']['digit_mean_tv'] for r in between['stable']]))
    report=dict(all_runs_complete=True,completed_training_runs=8,main_results=results,
                detached_negative_control=negative,between_seed_prediction_distances=between,
                paired_direct_minus_stable_summary=summary,
                digit_formula_mean_tv=within_mean,stable_between_seed_mean_tv=baseline,
                descriptive_tv_ratio=None if baseline==0 else within_mean/baseline,
                interpretation='Three seeds; descriptive pilot, no significance or general superiority claim',
                freeze_sha256=file_hash(args.out_dir/'freeze.json'))
    write_json(args.out_dir/'analysis.json',report)
    print(json.dumps(dict(status='complete',runs=8,analysis=str(args.out_dir/'analysis.json'))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--gate',type=Path,default=DEFAULT_OUT/'preflight.json')
    args=parser.parse_args()
    if args.action=='freeze':freeze(args)
    else:run(args)


if __name__=='__main__':main()
