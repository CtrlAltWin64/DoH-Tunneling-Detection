"""Reproducible preliminary real-capture study. Run from workspace root."""
import sys, json, pickle, hashlib, logging, argparse, shutil
from pathlib import Path
from dataclasses import asdict
from collections import Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
import numpy as np
import pandas as pd
import torch
from config import Config
from src.preprocessing import build_flows_from_pcap
from src.dataset_split import split_flows, split_train_calibration, create_split_manifest
from src.prefix_generation import build_prefix_table
from src.train_baseline import fit_rf, raw_proba
from src.calibration import fit_calibrator
from src.pipeline import environment_record
from src.adaptive_detector import curves_from_prefix_table, select_threshold, policy_outcomes

CFG = Config(checkpoints=(10,20), group_by='flow', model_type='both', pcap_label_rules={'benign':'benign','exfiltration':'malicious'})
torch.set_num_threads(4)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def prepare():
    if (OUT/'processed/flows.pkl').exists():
        raise FileExistsError('Preserve this completed dataset; copy scripts into a new results directory to rerun.')
    for d in ['tables','figures','metrics','predictions','models','processed','source_snapshot']:
        (OUT/d).mkdir(exist_ok=True, parents=True)
    dump('metrics/config.json',asdict(CFG)); dump('metrics/environment.json',environment_record())
    for p in [ROOT/'config.py', *sorted((ROOT/'src').glob('*.py'))]:
        dest=OUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    rows=[];flows=[];reports={}
    for p in sorted((ROOT/'data').glob('*.pcap*')):
        lab=0 if 'Benign' in p.name else 1
        stats=Counter(); print('Parsing',p.name,flush=True)
        fl=build_flows_from_pcap(p,lab,p.name,CFG,stats);flows.extend(fl);reports[p.name]=dict(stats)
        rows.append(dict(capture=p.name,label=lab,bytes=p.stat().st_size,sha256=sha(p),flows=len(fl)))
        print(p.name,len(fl),dict(stats),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics/capture_manifest.csv',index=False)
    dump('metrics/preprocessing.json',reports)
    with (OUT/'processed/flows.pkl').open('wb') as f:pickle.dump(flows,f)
    meta=pd.DataFrame([dict(flow_id=f.flow_id,label=f.label,source_file=f.source_file,stored_packets=len(f.ts),total_packets=f.total_packets,first_timestamp=float(f.ts[0]),last_timestamp=float(f.ts[-1]),finite_timestamps=bool(np.isfinite(f.ts).all()),monotonic_timestamps=bool((np.diff(f.ts)>=0).all())) for f in flows])
    meta.to_csv(OUT/'processed/flow_metadata.csv',index=False)
    counts=[]
    for source,g in [('ALL',meta),*list(meta.groupby('source_file'))]:
        for c in [10,20,30,40]:
            for lab in [0,1]:
                a=g[g.label==lab];counts.append(dict(capture=source,checkpoint=c,label=lab,total=len(a),eligible=int((a.stored_packets>=c).sum()),excluded=int((a.stored_packets<c).sum())))
    pd.DataFrame(counts).to_csv(OUT/'tables/checkpoint_eligibility.csv',index=False)
    usable=meta[meta.stored_packets>=10].copy()
    try:split_flows(usable,'source_file',42)
    except ValueError as e:dump('metrics/capture_split_blocker.json',{'error':str(e),'benign_captures':4,'malicious_captures':3})
    assign=split_flows(usable,'flow',42);usable['partition']=usable.flow_id.map(assign)
    usable.to_csv(OUT/'predictions/split_assignment.csv',index=False)
    dump('metrics/split_manifest.json',create_split_manifest(usable,assign,'flow',seed=42))

def train(which):
    if (OUT/f'models/{which}_bundle.pkl').exists():
        raise FileExistsError('Preserve the existing fitted model; use a new results directory to retrain.')
    with (OUT/'processed/flows.pkl').open('rb') as f: flows=pickle.load(f)
    split=pd.read_csv(OUT/'predictions/split_assignment.csv'); assign=dict(zip(split.flow_id,split.partition))
    parts={p:[f for f in flows if assign.get(f.flow_id)==p] for p in ['train','val','test']}
    cps=list(CFG.checkpoints)
    if which=='rf':
        print('Building RF prefixes',flush=True)
        tables={p:build_prefix_table(fl,cps) for p,fl in parts.items()}
        model=fit_rf(tables['train'],CFG);print('Fitting OOF calibration',flush=True)
        cal=fit_calibrator(tables['train'],CFG)
        with (OUT/'models/rf_bundle.pkl').open('wb') as f:pickle.dump((model,cal),f)
        scores={p:(tables[p],raw_proba(model,tables[p])) for p in ['val','test']}
    else:
        from src.cnn_gru_model import train_cnn_gru
        from src.cnn_gru_calibration import fit_calibrator_cnn
        from src.cnn_gru_prefix import build_cnn_prefix_table,raw_proba_cnn
        tm=split[split.partition=='train'].copy()
        ca=split_train_calibration(tm,'flow',42,.25)
        fitmeta=tm[tm.flow_id.map(ca)=='train_fit']
        ea=split_train_calibration(fitmeta,'flow',43,.25)
        roles={fid:('calibration' if ca[fid]=='train_cal' else ('early_stopping' if ea[fid]=='train_cal' else 'fit')) for fid in ca}
        tm['role']=tm.flow_id.map(roles);tm.to_csv(OUT/'predictions/cnn_training_roles.csv',index=False)
        inner={r:[f for f in parts['train'] if roles[f.flow_id]==r] for r in ['fit','calibration','early_stopping']}
        model,norm,hist=train_cnn_gru(inner['fit'],inner['early_stopping'],cps,CFG,OUT/'models/cnn_best.pt',device='cpu')
        dump('metrics/cnn_history.json',hist)
        cal,_=fit_calibrator_cnn(inner['calibration'],cps,model,norm,CFG,device='cpu')
        with (OUT/'models/cnn_bundle.pkl').open('wb') as f:pickle.dump((model,norm,cal),f)
        scores={}
        for p in ['val','test']:
            table=build_cnn_prefix_table(parts[p],cps,model,norm);scores[p]=(table,raw_proba_cnn(table))
    dump(f'metrics/{which}_calibration.json',cal.describe())
    curves={}
    for p,(tab,raw) in scores.items():
        prob=cal.transform(raw,tab.checkpoint.to_numpy()); tab=tab.copy();tab['raw_probability']=raw;tab['probability']=prob
        tab.to_csv(OUT/f'predictions/{which}_{p}_prefixes.csv',index=False)
        curves[p]=curves_from_prefix_table(tab,prob,cps)
    thresholds={}
    for policy,idx in [('adaptive',None),('fixed_10',[0]),('fixed_20',[1])]:
        selected=select_threshold(curves['val'],.01,.001,idx)
        selected.pop('sweep').to_csv(OUT/f'metrics/{which}_{policy}_threshold_sweep.csv',index=False)
        thresholds[policy]=selected
        for p in ['val','test']:
            policy_outcomes(curves[p],selected['tau'],idx).to_csv(OUT/f'predictions/{which}_{p}_{policy}_trace.csv',index=False)
    dump(f'metrics/{which}_thresholds.json',thresholds)
    print(which,'COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','rf','cnn']);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    else:train(args.stage)
