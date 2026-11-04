"""Frozen-source paired seed extension; canonical arithmetic, exhaustive records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
for _thread_key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_key]='1'
import numpy as np

CORE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(CORE))
from context_study import atomic_json
from evaluate import load,make_set,run_model,option_metrics,noul_metrics
from reproduce_training import weight_hash


def save_predictions(path, records):
    arrays={}
    for typ in ('choice','score','noul'):
        rows=[r for r in records if r['type']==typ]
        arrays[typ+'_targets']=np.array([r['y'] for r in rows])
        if typ=='noul':
            arrays['noul_logits']=np.array([r['logit'] for r in rows])
            arrays['noul_bayes']=np.array([r['bayes'] for r in rows])
        else:
            n=np.array([len(r['scores']) for r in rows])
            scores=np.full((len(rows),int(n.max())),-np.inf)
            bayes=np.zeros_like(scores)
            for i,r in enumerate(rows):
                scores[i,:n[i]]=r['scores'];bayes[i,:n[i]]=r['bayes']
            arrays[typ+'_scores']=scores;arrays[typ+'_n']=n;arrays[typ+'_bayes']=bayes
    np.savez_compressed(path,**arrays)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--formula',choices=['stable','original'],required=True)
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-seed',type=int,default=5001)
    p.add_argument('--validation-requests',type=int,default=2000)
    p.add_argument('--test-seed',type=int,default=5002)
    p.add_argument('--test-requests',type=int,default=4000)
    p.add_argument('--skip-evaluation',action='store_true',help='Training-only smoke run; no held-out data is generated')
    args=p.parse_args()
    if args.steps<1 or args.validation_requests<1 or args.test_requests<1:
        p.error('steps and evaluation request counts must be positive')
    output=args.output.resolve()
    if (output/'results.json').exists():raise FileExistsError('Completed output exists')
    output.mkdir(parents=True,exist_ok=True)
    source=output/'source';source.mkdir(exist_ok=True)
    hashes={}
    for name in ('autograd.py','data.py','model.py','train.py'):
        data=(CORE/name).read_bytes();(source/name).write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    checkpoints=output/'checkpoints';checkpoints.mkdir(exist_ok=True)
    config={'seed':args.seed,'formula':args.formula,'steps':args.steps,'batch':32,
            'role':'Exploratory paired seed extension; seed0 seen previously',
            'parameter_dtype':'float32','target_dtype':'float64 (canonical)',
            'source_sha256':hashes,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'test_seed':args.test_seed,'test_requests':args.test_requests,
            'validation_seed':args.validation_seed,'validation_requests':args.validation_requests,
            'skip_evaluation':args.skip_evaluation,
            'numpy':np.__version__,'python':sys.version}
    atomic_json(output/'config.json',config)
    launcher=r'''
import hashlib,json,runpy,sys,os
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1])
import data as D
import autograd as ag
from model import TinyS1
if sys.argv[4]=='original':ag.sigmoid_np=lambda z:1./(1.+np.exp(-z))
output=Path(sys.argv[2]);seed=sys.argv[3];steps=sys.argv[5]
original=D.collate;constructor=TinyS1.__init__;opt_constructor=ag.AdamW.__init__
seen=[];optimizers=[];canonical=hashlib.sha256();complete=hashlib.sha256()
def init(self,*args,**kw):
    constructor(self,*args,**kw);h=hashlib.sha256()
    for param in self.parameters():h.update(str(param.shape).encode());h.update(param.data.tobytes())
    seen.append(h.hexdigest())
TinyS1.__init__=init
def optimizer_init(self,*args,**kw):
    opt_constructor(self,*args,**kw);optimizers.append(self)
ag.AdamW.__init__=optimizer_init
def batch(reqs):
    value=original(reqs)
    for h,names in ((canonical,('ids','pos','seg','opt_rows','opt_group','opt_target')),
                    (complete,('ids','pos','seg','opt_rows','opt_group','opt_target','noul_rows','noul_target','mask_bias'))):
        for name in names:
            a=np.ascontiguousarray(value[name]);h.update((name+str(a.shape)+str(a.dtype)).encode());h.update(a.tobytes())
    return value
D.collate=batch
sys.argv=[str(Path(sys.argv[1])/'train.py'),'--run','stream','--steps',steps,'--batch','32','--seed',seed,'--output-dir',str(output/'checkpoints')]
runpy.run_path(sys.argv[0],run_name='__main__')
opt=optimizers[0]
np.savez_compressed(output/'optimizer-final.npz',**{**{'m_'+str(i):a for i,a in enumerate(opt.m)},**{'v_'+str(i):a for i,a in enumerate(opt.v)}})
(output/'execution.json').write_text(json.dumps({'initial_parameter_sha256':seen[0],
 'training_data_sha256':canonical.hexdigest(),'complete_batch_sha256':complete.hexdigest(),
 'optimizer_steps':opt.t,'optimizer_betas':opt.betas,'optimizer_epsilon':opt.eps,
 'final_lr':opt.lr,'numpy':np.__version__,'python':sys.version},indent=2))
'''
    env=os.environ.copy()
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    start=time.perf_counter()
    subprocess.run([sys.executable,'-c',launcher,str(source),str(output),str(args.seed),args.formula,str(args.steps)],
                   cwd=checkpoints,env=env,check=True,stdout=sys.stdout,stderr=sys.stderr,
                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    model=load('stream',checkpoints)
    evaluations={}
    for split,seed,n in (() if args.skip_evaluation else
                         (('validation',args.validation_seed,args.validation_requests),
                          ('test',args.test_seed,args.test_requests))):
        records=run_model(model,make_set(seed,n))
        save_predictions(output/(split+'-predictions.npz'),records)
        evaluations[split]={'choice':option_metrics(records,'choice'),'score':option_metrics(records,'score'),
                            'noul':noul_metrics(records)}
    result={'config':config,'execution':json.loads((output/'execution.json').read_text()),
            'evaluations':evaluations,'weight_data_sha256':weight_hash(checkpoints/'ckpt_stream.npz'),
            'total_seconds':time.perf_counter()-start}
    atomic_json(output/'results.json',result)
    print(json.dumps({'complete':True,'seed':args.seed,'formula':args.formula,
                      'choice_accuracy':None if args.skip_evaluation else evaluations['test']['choice']['acc']}),flush=True)


if __name__=='__main__':main()
