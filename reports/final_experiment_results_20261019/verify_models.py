"""Reload persisted models, replay saved scores, and verify trace timing from flows."""
import sys,json,pickle,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from src.train_baseline import raw_proba
from src.cnn_gru_model import score_flows_at_checkpoint
torch.set_num_threads(4)
checks=[]
def check(name,passed,**detail):
    checks.append(dict(check=name,passed=bool(passed),**detail));assert passed,name
with (OUT/'processed/flows.pkl').open('rb') as f:flows=pickle.load(f)
fm={f.flow_id:f for f in flows};split=pd.read_csv(OUT/'predictions/split_assignment.csv');assign=dict(zip(split.flow_id,split.partition))
for model in ['rf','cnn']:
    with (OUT/f'models/{model}_bundle.pkl').open('rb') as f:bundle=pickle.load(f)
    for part in ['val','test']:
        pred=pd.read_csv(OUT/f'predictions/{model}_{part}_prefixes.csv')
        if model=='rf':
            fitted,cal=bundle;raw=raw_proba(fitted,pred)
        else:
            fitted,norm,cal=bundle;fl=[f for f in flows if assign.get(f.flow_id)==part];lookup={}
            for cp in [10,20]:
                probabilities,ids=score_flows_at_checkpoint(fl,cp,fitted,norm)
                lookup.update({(fid,cp):p for fid,p in zip(ids,probabilities)})
            raw=np.array([lookup[fid,cp] for fid,cp in zip(pred.flow_id,pred.checkpoint)])
        prob=cal.transform(raw,pred.checkpoint.to_numpy());delta=float(np.max(np.abs(prob-pred.probability)))
        check(model+part+'_model_reload',np.allclose(prob,pred.probability,atol=1e-7,rtol=1e-6),max_probability_difference=delta)
        for policy in ['adaptive','fixed_10','fixed_20']:
            trace=pd.read_csv(OUT/f'predictions/{model}_{part}_{policy}_trace.csv');lat=[]
            for r in trace.itertuples():
                flow=fm[r.flow_id];cp=int(r.observed_packets)
                lat.append(float(flow.ts[cp-1]-flow.ts[0]) if r.alert else np.nan)
            check(model+part+policy+'_latency',np.allclose(lat,trace.latency_s,atol=1e-9,rtol=1e-9,equal_nan=True))
signatures=defaultdict(set);counts=Counter();rows=[]
for f in flows:
    if f.flow_id in assign and len(f.ts)>=20:
        sig=(tuple(f.length[:20]),tuple(f.direction[:20]));signatures[sig].add(assign[f.flow_id]);counts[sig]+=1
for lab in [0,1]:
    fs=[f for f in flows if f.label==lab];lengths=Counter(int(f.length[0]) for f in fs)
    for size,n in sorted(lengths.items()):rows.append(dict(label=lab,first_packet_ip_length=size,flows=n))
pd.DataFrame(rows).to_csv(OUT/'tables/first_packet_length_distribution.csv',index=False)
confounds=dict(common_prefix_signatures_shared_train_test=sum({'train','test'}<=s for s in signatures.values()),flows_in_repeated_common_prefix_signatures=sum(n for n in counts.values() if n>1),note='Length/direction signatures omit timing. Repetition is a dependence/confounding diagnostic, not proof of duplicate capture content.')
(OUT/'metrics/confounding_diagnostics.json').write_text(json.dumps(confounds,indent=2))
(OUT/'metrics/model_replay_verification.json').write_text(json.dumps(dict(checks=checks,passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks)),indent=2))
hashes={}
for p in sorted(OUT.rglob('*')):
    if p.is_file() and 'test_temp' not in p.parts and '__pycache__' not in p.parts and p.name!='artifact_hashes.json' and p.suffix not in ['.log']:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        hashes[str(p.relative_to(OUT))]=h.hexdigest()
(OUT/'metrics/artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(dict(checks=checks,confounds=confounds,hashed_artifacts=len(hashes)),indent=2))
