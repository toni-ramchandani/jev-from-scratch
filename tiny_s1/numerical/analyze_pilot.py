"""Per-seed proper scores and paired functional comparisons; no tuning."""
import argparse
import os
from pathlib import Path
from itertools import combinations
import csv
import hashlib
import json
for _thread_key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_key]='1'
import numpy as np

CORE = Path(__file__).resolve().parent.parent

def probabilities(a, head):
    if head == 'noul':
        z = a['noul_logits'].astype(np.float64)
        e = np.exp(-np.abs(z))
        p = np.where(z >= 0, 1/(1+e), e/(1+e))
        return np.column_stack((1-p,p))
    z = a[head+'_scores'].astype(np.float64)
    e = np.exp(z-z.max(axis=1,keepdims=True))
    return e/e.sum(axis=1,keepdims=True)

def ece(confidence, correctness):
    total = 0.
    for lo, hi in zip(np.linspace(0,1,11)[:-1], np.linspace(0,1,11)[1:]):
        mask = (confidence>=lo if lo==0 else confidence>lo) & (confidence<=hi)
        if mask.any():
            total += mask.mean()*abs(confidence[mask].mean()-correctness[mask].mean())
    return float(total)

def metrics(a, head):
    p = probabilities(a,head)
    y = a[head+'_targets'].astype(np.int64)
    pred = p.argmax(axis=1)
    if head == 'noul':
        z = a['noul_logits'].astype(float)
        nll = np.mean(np.logaddexp(0,z)-y*z)
        brier = np.mean((p[:,1]-y)**2)
        calibration = ece(p[:,1],y)
    else:
        z = a[head+'_scores'].astype(float)
        centered = z-z.max(axis=1,keepdims=True)
        nll = np.mean(np.log(np.exp(centered).sum(axis=1))-centered[np.arange(len(y)),y])
        onehot = np.eye(p.shape[1])[y]
        brier = np.mean(np.sum((p-onehot)**2,axis=1))
        calibration = ece(p.max(axis=1),(pred==y).astype(float))
    out = {'n':len(y),'acc':float(np.mean(pred==y)),'nll':float(nll),
           'brier':float(brier),'ece':calibration}
    if head == 'choice':
        accepted = p.max(axis=1)>=.8
        out['threshold_0_8']={'accepted':int(accepted.sum()),'coverage':float(accepted.mean()),
             'observed_error_rate':float(np.mean(pred[accepted]!=y[accepted])) if accepted.any() else None}
    if head == 'score':
        expectation = p@np.arange(p.shape[1])
        out['expected_index_mae_to_observed_label'] = float(np.mean(np.abs(expectation-y)))
    return out

def compare(a,b,head):
    assert np.array_equal(a[head+'_targets'],b[head+'_targets'])
    assert np.array_equal(a[head+'_bayes'],b[head+'_bayes'])
    if head != 'noul':
        assert np.array_equal(a[head+'_n'],b[head+'_n'])
    p,q = probabilities(a,head),probabilities(b,head)
    assert p.shape==q.shape
    tv = .5*np.abs(p-q).sum(axis=1)
    result = {'n':len(p),'mean_tv':float(tv.mean()),'max_tv':float(tv.max()),
              'max_probability_difference':float(np.max(np.abs(p-q))),
              'argmax_disagreement_count':int(np.sum(p.argmax(axis=1)!=q.argmax(axis=1))),
              'argmax_disagreement_rate':float(np.mean(p.argmax(axis=1)!=q.argmax(axis=1)))}
    if head == 'choice':
        accept_p, accept_q = p.max(axis=1)>=.8,q.max(axis=1)>=.8
        action_p = np.where(accept_p,p.argmax(axis=1),-1)
        action_q = np.where(accept_q,q.argmax(axis=1),-1)
        result['threshold_0_8']={
            'acceptance_disagreement_count':int(np.sum(accept_p!=accept_q)),
            'acceptance_disagreement_rate':float(np.mean(accept_p!=accept_q)),
            'full_action_disagreement_count':int(np.sum(action_p!=action_q)),
            'full_action_disagreement_rate':float(np.mean(action_p!=action_q))}
    if head == 'score':
        result['mean_expected_index_absolute_difference']=float(np.mean(np.abs((p-q)@np.arange(p.shape[1]))))
    return result

def summarize(values):
    a = np.asarray(values,dtype=float)
    return {'n_training_seeds':len(a),'mean':float(a.mean()),
            'sample_sd':float(a.std(ddof=1)) if len(a)>1 else None,
            'minimum':float(a.min()),'maximum':float(a.max())}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True,help='Directory containing runs/seed-{0,1,2}/{stable,original}')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    data_root=args.data_root.resolve()
    output_dir=args.output_dir.resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    heads = ('choice','score','noul')
    runs, arrays, rows = {},{},[]
    for seed in (0,1,2):
        for formula in ('stable','original'):
            path = data_root/'runs'/f'seed-{seed}'/formula
            data = json.loads((path/'results.json').read_text())
            assert data['config']['steps']==4000
            assert data['config']['seed']==seed and data['config']['formula']==formula
            assert data['config']['test_seed']==5002 and data['config']['test_requests']==4000
            if seed>0:
                assert data['config']['validation_seed']==5001 and data['config']['validation_requests']==2000
            for name,digest in data['config']['source_sha256'].items():
                assert hashlib.sha256((path/'source'/name).read_bytes()).hexdigest()==digest, (seed,formula,name)
            record = {'seed':seed,'formula':formula,'weight_data_sha256':data['weight_data_sha256'],
                      'execution':data['execution'],'splits':{},
                      'prediction_file_sha256':{split:hashlib.sha256((path/(split+'-predictions.npz')).read_bytes()).hexdigest()
                                                for split in ('validation','test')}}
            for split in ('validation','test'):
                a = dict(np.load(path/(split+'-predictions.npz'),allow_pickle=False))
                arrays[seed,formula,split]=a
                record['splits'][split]={head:metrics(a,head) for head in heads}
                for head in heads:
                    for key in ('nll','brier','ece','acc'):
                        # Legacy evaluator clips binary p for Brier/ECE only; this
                        # analysis consistently evaluates unmodified probabilities.
                        tolerance=2.1e-12 if head=='noul' and key in ('brier','ece') else 1e-12
                        assert abs(record['splits'][split][head][key]-data['evaluations'][split][head][key])<tolerance
                    rows.append({'seed':seed,'formula':formula,'split':split,'head':head,
                                 **{k:record['splits'][split][head][k] for k in ('n','acc','nll','brier','ece')}})
            runs[seed,formula]=record
    paired=[]
    for seed in (0,1,2):
        s,o=runs[seed,'stable'],runs[seed,'original']
        for key in ('initial_parameter_sha256','training_data_sha256'):
            assert s['execution'][key]==o['execution'][key], (seed,key)
        if seed>0:
            assert s['execution']['complete_batch_sha256']==o['execution']['complete_batch_sha256']
            assert s['execution']['optimizer_steps']==o['execution']['optimizer_steps']==4000
        record={'seed':seed,'split':'test','heads':{}}
        for head in heads:
            delta={key:s['splits']['test'][head][key]-o['splits']['test'][head][key]
                   for key in ('acc','nll','brier','ece')}
            record['heads'][head]={'stable_minus_original':delta,
                'functional_difference':compare(arrays[seed,'stable','test'],arrays[seed,'original','test'],head)}
        paired.append(record)
    ordinary=[]
    for formula in ('stable','original'):
        for a,b in combinations((0,1,2),2):
            ordinary.append({'formula':formula,'seeds':[a,b],
                'heads':{head:compare(arrays[a,formula,'test'],arrays[b,formula,'test'],head) for head in heads}})
    summary={head:{
        'signed_paired_delta':{key:summarize([r['heads'][head]['stable_minus_original'][key] for r in paired])
                               for key in ('acc','nll','brier','ece')},
        'within_seed_implementation_variation':{key:summarize([r['heads'][head]['functional_difference'][key] for r in paired])
                                   for key in ('mean_tv','argmax_disagreement_rate')},
        'between_seed_variation':{formula:{key:{'n_seed_pairs':3,
                 'mean':float(np.mean([r['heads'][head][key] for r in ordinary if r['formula']==formula]))}
                 for key in ('mean_tv','argmax_disagreement_rate')} for formula in ('stable','original')}
        } for head in heads}
    result={'scope':'Exploratory pilot; three training seeds, previously inspected test seed5002; no confirmatory significance claim',
            'delta_convention':'stable minus original; negative NLL/Brier is better for stable',
            'brier_convention':'sum over categories for Choice/Score; one Bernoulli coordinate for Noul',
            'probability_evaluation':'Common float64 stable evaluation, without clipping; old binary Brier/ECE may differ by at most2.1e-12 due to legacy clipping',
            'threshold':'Fixed 0.8 maximum Choice probability; diagnostic, not a calibrated risk guarantee',
            'ordinary_seed_comparisons':'Three correlated seed pairs per formula; means are descriptive, not six independent replicates',
            'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runs':list(runs.values()),'paired_comparisons':paired,'ordinary_seed_comparisons_detail':ordinary,
            'summary':summary}
    (output_dir/'pilot-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output_dir/'per-seed-metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
