"""Fetch/verify pinned original UCI optical-digits files, preserving official splits."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT_DATA=HERE.parents[2].parent/'digits-numerical-output/data'


def verify_bytes(name, payload, info):
    assert len(payload)==info['bytes'], (name,'length mismatch')
    assert hashlib.sha256(payload).hexdigest()==info['sha256'], (name,'SHA256 mismatch')
    if 'shape' in info:
        raw=np.loadtxt(io.BytesIO(payload),delimiter=',',dtype=np.int64)
        assert list(raw.shape)==info['shape'], (name,'shape mismatch')
        assert np.all((raw[:,:64]>=0)&(raw[:,:64]<=16))
        assert np.all((raw[:,64]>=0)&(raw[:,64]<=9))
        assert np.bincount(raw[:,64],minlength=10).tolist()==info['class_counts']
        canonical=np.ascontiguousarray(raw,dtype='<i8')
        assert hashlib.sha256(canonical.tobytes()).hexdigest()==info['parsed_int64_sha256']
        x=raw[:,:64].astype(np.float32)/np.float32(16)
        parity=(raw[:,64]%2).astype(np.float32)
        assert x.dtype==np.float32 and parity.dtype==np.float32
        assert np.all(np.isfinite(x))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--verify-only',action='store_true',help='Read local files only; no network or writes')
    args=parser.parse_args()
    dataset=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))['dataset']
    payloads={}
    for name,info in dataset['files'].items():
        target=args.data_dir/name
        if target.exists():
            payload=target.read_bytes()
        else:
            assert not args.verify_only, f'Missing local file: {target}'
            assert info['url'].startswith('https://archive.ics.uci.edu/'), 'Only the official UCI host is allowed'
            with urlopen(info['url'],timeout=60) as response:
                assert response.url.startswith('https://archive.ics.uci.edu/'), 'Unexpected redirect host'
                payload=response.read()
        verify_bytes(name,payload,info)
        payloads[name]=payload
    if not args.verify_only:
        args.data_dir.mkdir(parents=True,exist_ok=True)
        for name,payload in payloads.items():
            target=args.data_dir/name
            if not target.exists():target.write_bytes(payload)
        metadata=dict(dataset,verified_at_utc=datetime.now(timezone.utc).isoformat(),
                      transformations='Raw files unchanged; FP32 pixels /16 and odd-parity labels derived at training')
        (args.data_dir/'dataset-manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(verified=True,files=list(payloads),data_dir=str(args.data_dir),
                          license=dataset['license'],doi=dataset['doi'],verify_only=args.verify_only)))


if __name__=='__main__':main()
