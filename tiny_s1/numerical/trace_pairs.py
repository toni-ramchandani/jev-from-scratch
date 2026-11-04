"""Paired arithmetic diagnostics; no held-out test evaluation or core mutation."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(SOURCE))
import autograd as ag
import data as D
from model import TinyS1
from train import lr_at

STABLE=ag.sigmoid_np
ANCHORS={0,1,2,9,24,49,99,199}
MODES=('canonical_fp32','typed_targets_fp32','fp64_from_fp32','detached_bce_fp32')


def original(z):
    return 1.0/(1.0+np.exp(-z))


def names(model):
    result=['tok','pos','seg','lnf_g','lnf_b','w_opt','w_noul','b_noul']
    result += [f'layer.{i}.{key}' for i,layer in enumerate(model.layers) for key in layer]
    return result


def fingerprint(arrays, quantize=None):
    digest=hashlib.sha256()
    for array in arrays:
        a=np.asarray(array,dtype=quantize) if quantize is not None else array
        digest.update(str(a.shape).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def update_batch_hash(digest,batch):
    for name in ('ids','pos','seg','opt_rows','opt_group','opt_target','noul_rows','noul_target','mask_bias'):
        a=np.ascontiguousarray(batch[name])
        digest.update((name+str(a.shape)+str(a.dtype)).encode());digest.update(a.tobytes())


def diff(a,b):
    if a is None or b is None:
        return {'missing_a':a is None,'missing_b':b is None}
    x,y=a.astype(float),b.astype(float)
    delta=x-y;norm_a=float(np.linalg.norm(x));norm_b=float(np.linalg.norm(y))
    size=float(np.linalg.norm(delta))
    return {'max_abs':float(np.max(np.abs(delta))) if delta.size else 0.,
            'l2':size,'a_l2':norm_a,'b_l2':norm_b,
            'relative_l2':size/max(norm_a,norm_b,1e-300),
            'cosine':float(np.sum(x*y)/(norm_a*norm_b)) if norm_a and norm_b else None,
            'different_values':int(np.count_nonzero(a!=b)),'dtype_a':str(a.dtype),'dtype_b':str(b.dtype)}


def compare(first,second,labels,detail=False,d=64):
    blocks={label:diff(a,b) for label,a,b in zip(labels,first,second)}
    available=[(label,row) for label,row in blocks.items() if 'max_abs' in row]
    result={'max_abs':max((row['max_abs'] for _,row in available),default=0.),
            'l2':float(np.sqrt(sum(row['l2']**2 for _,row in available))),
            'different_blocks':[label for label,row in available if row['different_values']],
            'different_values':sum(row['different_values'] for _,row in available),
            'dtype_pairs':sorted({row['dtype_a']+'/'+row['dtype_b'] for _,row in available})}
    shared=[row for label,row in available if label not in ('w_noul','b_noul')]
    result['shared_max_abs']=max((row['max_abs'] for row in shared),default=0.)
    result['shared_l2']=float(np.sqrt(sum(row['l2']**2 for row in shared)))
    if detail:
        result['blocks']=blocks
        result['qkv_bias_slices']={}
        for label,a,b in zip(labels,first,second):
            if label.endswith('.b_qkv'):
                for index,kind in enumerate(('q','k','v')):
                    result['qkv_bias_slices'][label+'.'+kind]=diff(a[index*d:(index+1)*d],b[index*d:(index+1)*d])
    return result


def forward_loss(model,batch,detach=False):
    if not detach:
        return model.loss(batch)
    hidden=model.encode(batch)
    scores=(ag.take_rows(hidden,batch['opt_rows'])@model.w_opt).reshape(-1)
    noul_hidden=ag.Tensor(ag.take_rows(hidden,batch['noul_rows']).data)
    logits=(noul_hidden@model.w_noul+model.b_noul).reshape(-1)
    return (ag.grouped_softmax_xent(scores,batch['opt_group'],batch['opt_target'],batch['n_groups'])
            +ag.bce_with_logits(logits,batch['noul_target'])),(scores,logits)


def gradient_pass(model,batch,formula,detach=False):
    for p in model.parameters():p.grad=None
    ag.sigmoid_np=STABLE if formula=='stable' else original
    loss,(scores,logits)=forward_loss(model,batch,detach)
    loss.backward()
    grads=[None if p.grad is None else p.grad.copy() for p in model.parameters()]
    return {'loss':float(loss.data),'scores':scores.data.copy(),'logits':logits.data.copy(),'gradients':grads}


def scalar_error(logits):
    a,b=STABLE(logits),original(logits)
    z=logits.astype(np.float64)
    e=np.exp(-np.abs(z));reference=np.where(z>=0,1./(1.+e),e/(1.+e))
    unsigned=np.uint32 if logits.dtype==np.float32 else np.uint64
    ua,ub=np.ascontiguousarray(a).view(unsigned),np.ascontiguousarray(b).view(unsigned)
    ulps=np.maximum(ua,ub)-np.minimum(ua,ub)
    return {'logit_min':float(z.min()),'logit_max':float(z.max()),
            'dtype':str(logits.dtype),'formula_difference':diff(a,b),
            'formula_max_ulp':int(ulps.max()),
            'stable_max_abs_to_float64':float(np.abs(a.astype(float)-reference).max()),
            'original_max_abs_to_float64':float(np.abs(b.astype(float)-reference).max())}


def atomic_json(path,data):
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def run(args):
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'results.json').exists():raise FileExistsError('Completed trace already exists')
    ag.set_dtype(np.float32)
    base=TinyS1(len(D.VOCAB),seed=args.seed)
    initial=[p.data.copy() for p in base.parameters()]
    dtype=np.float64 if args.mode=='fp64_from_fp32' else np.float32
    ag.set_dtype(dtype)
    models=[TinyS1(len(D.VOCAB),seed=args.seed) for _ in range(2)]
    for model in models:
        for p,value in zip(model.parameters(),initial):p.data[:]=value.astype(dtype)
    labels=names(models[0]);optimizers=[ag.AdamW(m.parameters(),decay=m.matrices()) for m in models]
    config={'mode':args.mode,'seed':args.seed,'steps':args.steps,'batch':args.batch,
            'scheduled_training_steps':4000,'parameter_dtype':str(np.dtype(dtype)),
            'initial_parameter_sha256':fingerprint([p.data for p in models[0].parameters()]),
            'initial_fp32_values_sha256':fingerprint(initial), 'numpy':np.__version__,'python':sys.version,
            'detached_binary_gradient':args.mode=='detached_bce_fp32',
            'cast_targets_to_parameter_dtype':args.mode=='typed_targets_fp32',
            'source_sha256':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                             for name in ('autograd.py','data.py','model.py','train.py')},
            'trace_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(output/'config.json',config)
    canonical_hash=hashlib.sha256();effective_hash=hashlib.sha256();rng=D.make_rng(1000+args.seed)
    records=[];anchors=[];first_gradient=None;first_parameter=None;first_shared=None;began=time.perf_counter()
    for step in range(args.steps):
        batch=D.collate([D.sample_request(rng) for _ in range(args.batch)])
        update_batch_hash(canonical_hash,batch)
        if args.mode=='typed_targets_fp32':
            batch=dict(batch,opt_target=batch['opt_target'].astype(dtype),noul_target=batch['noul_target'].astype(dtype))
        update_batch_hash(effective_hash,batch)
        before=[[p.data.copy() for p in model.parameters()] for model in models]
        actual=[gradient_pass(model,batch,formula,config['detached_binary_gradient'])
                for model,formula in zip(models,('stable','original'))]
        grads=compare(actual[0]['gradients'],actual[1]['gradients'],labels,step in ANCHORS)
        if first_gradient is None and grads['different_values']:first_gradient=step
        if step in ANCHORS or step==args.steps-1:
            cf_direct=gradient_pass(models[0],batch,'original',config['detached_binary_gradient'])
            cf_stable=gradient_pass(models[1],batch,'stable',config['detached_binary_gradient'])
            anchors.append({'step':step,
                'same_stable_state':{'gradient_difference':compare(actual[0]['gradients'],cf_direct['gradients'],labels,True),
                                     'scalar_sigmoid_error':scalar_error(actual[0]['logits'])},
                'same_original_state':{'gradient_difference':compare(cf_stable['gradients'],actual[1]['gradients'],labels,True),
                                       'scalar_sigmoid_error':scalar_error(actual[1]['logits'])},
                'cross_trajectory_gradients':grads})
        for model,optimizer,item in zip(models,optimizers,actual):
            for p,gradient in zip(model.parameters(),item['gradients']):p.grad=gradient
            optimizer.lr=lr_at(step,4000)
            optimizer.step()
        after=[[p.data.copy() for p in model.parameters()] for model in models]
        parameter_diff=compare(after[0],after[1],labels,step in ANCHORS)
        if first_parameter is None and parameter_diff['different_values']:first_parameter=step
        if first_shared is None and parameter_diff['shared_max_abs']>0:first_shared=step
        updates=[[new-old for new,old in zip(end,start)] for end,start in zip(after,before)]
        entry={'step':step,'stable_loss':actual[0]['loss'],'original_loss':actual[1]['loss'],
               'option_logit_difference':diff(actual[0]['scores'],actual[1]['scores']),
               'binary_logit_difference':diff(actual[0]['logits'],actual[1]['logits']),
               'gradient_difference':grads,'parameter_difference':parameter_diff,
               'update_difference':compare(updates[0],updates[1],labels,step in ANCHORS),
               'gradient_dtypes_stable':dict(zip(labels,[str(g.dtype) if g is not None else None for g in actual[0]['gradients']]))}
        records.append(entry)
        if step%25==0 or step==args.steps-1:
            progress={'step':step,'mode':args.mode,'parameter_max_abs':parameter_diff['max_abs'],
                      'shared_max_abs':parameter_diff['shared_max_abs'],'seconds':time.perf_counter()-began}
            atomic_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    checkpoints={}
    for model,formula in zip(models,('stable','original')):
        path=output/(formula+'.npz');np.savez(path,*[p.data for p in model.parameters()])
        checkpoints[formula]={'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                              'array_sha256':fingerprint([p.data for p in model.parameters()])}
    result={'config':config,'canonical_training_data_sha256':canonical_hash.hexdigest(),
            'effective_training_data_sha256':effective_hash.hexdigest(),
            'first_gradient_difference_step':first_gradient,'first_parameter_difference_step':first_parameter,
            'first_shared_parameter_difference_step':first_shared,'steps':records,'same_state_anchors':anchors,
            'checkpoints':checkpoints,'final_parameters':records[-1]['parameter_difference'],
            'seconds':time.perf_counter()-began,'role':'Diagnostic trace; no test evaluation'}
    atomic_json(output/'results.json',result)
    if config['detached_binary_gradient'] and result['final_parameters']['shared_max_abs']!=0:
        raise AssertionError('Detached binary gradient still changed shared/categorical parameters')
    print(json.dumps({'complete':True,'mode':args.mode,'seconds':result['seconds'],
                      'first_shared_step':first_shared,'final_max_abs':result['final_parameters']['max_abs']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.steps<1 or args.batch<1:parser.error('steps and batch must be positive')
    run(args)
