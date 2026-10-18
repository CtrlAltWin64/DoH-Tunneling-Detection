"""Inspect complete PCAPNG structure and independently check saved synthetic scores."""
import json, struct, hashlib, sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
checks=[]
for p in sorted((ROOT/'data').glob('*.pcapng')):
    blocks=Counter();endians=set();links=set();offsets=[];invalid_interfaces=0;interfaces=[];endian='<'
    with p.open('rb') as f:
        while hdr:=f.read(8):
            assert len(hdr)==8
            if hdr[:4]==b'\x0a\x0d\x0d\x0a':
                bom=f.read(4);endian='<' if bom==b'\x4d\x3c\x2b\x1a' else '>'
                endians.add(endian);n=struct.unpack(endian+'I',hdr[4:])[0];body=bom+f.read(n-16);interfaces=[]
            else:
                n=struct.unpack(endian+'I',hdr[4:])[0];body=f.read(n-12)
            assert n>=12 and n%4==0 and len(body)==n-12
            assert struct.unpack(endian+'I',f.read(4))[0]==n
            typ=struct.unpack(endian+'I',hdr[:4])[0];blocks[typ]+=1
            if typ==1:
                lt=struct.unpack(endian+'H',body[:2])[0];links.add(lt);interfaces.append(lt);i=8
                while i+4<=len(body):
                    code,length=struct.unpack(endian+'HH',body[i:i+4]);value=body[i+4:i+4+length]
                    if code==0:break
                    if code==14:offsets.append(struct.unpack(endian+'q',value)[0])
                    i+=4+length+(-length)%4
            if typ==6:
                interface=struct.unpack(endian+'I',body[:4])[0]
                invalid_interfaces+=interface>=len(interfaces)
    checks.append(dict(capture=p.name,blocks=dict(blocks),endians=sorted(endians),linktypes=sorted(links),timestamp_offsets=offsets,invalid_interfaces=invalid_interfaces,latency_structure_supported=blocks[3]==0 and len(endians)==1 and links=={1} and not any(offsets) and invalid_interfaces==0))
(OUT/'metrics/pcapng_structure.json').write_text(json.dumps(checks,indent=2))
saved=ROOT/'data/results/synthetic_smoke_test';results=[]
required=[saved/f'{prefix}{name}' for prefix in ['', 'cnn_gru_']
          for name in ['thresholds.json','predictions_val.csv','predictions_test.csv']]
required += [saved/f'tables/{prefix}fixed_vs_adaptive_{part}.csv'
             for prefix in ['', 'cnn_gru_'] for part in ['validation','test']]
if not all(p.exists() for p in required):
    status={'status':'skipped','reason':'Original local synthetic artifacts are absent; no synthetic replay was performed.',
            'missing':[str(p.relative_to(ROOT)) for p in required if not p.exists()]}
    (OUT/'metrics/synthetic_replay_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps({'pcapng_checks':checks,'synthetic_replay':status},indent=2))
    sys.exit(0)
for model,prefix in [('rf',''),('cnn','cnn_gru_')]:
    thresholds=json.loads((saved/f'{prefix}thresholds.json').read_text())
    for part,suffix in [('val','validation'),('test','test')]:
        pred=pd.read_csv(saved/f'{prefix}predictions_{part}.csv');tab=pd.read_csv(saved/f'tables/{prefix}fixed_vs_adaptive_{suffix}.csv')
        cps=[int(c[2:]) for c in pred if c.startswith('p_')];y=pred.label.to_numpy()
        for r in tab.to_dict('records'):
            policy=r['policy'];tau=thresholds[policy]['tau'];cols=cps if policy=='adaptive' else [int(policy.split('_')[1])]
            P=pred[[f'p_{c}' for c in cols]].to_numpy();hit=P>=tau;alert=hit.any(axis=1);obs=np.array(cols)[np.where(alert,hit.argmax(axis=1),len(cols)-1)]
            actual={'TP':int(sum(alert&(y==1))),'FP':int(sum(alert&(y==0))),'TN':int(sum(~alert&(y==0))),'FN':int(sum(~alert&(y==1))),'avg_packets_all':float(obs.mean())}
            for k,v in actual.items():
                results.append(dict(model=model,partition=part,policy=policy,metric=k,recomputed=v,saved=r[k],passed=bool(np.isclose(v,r[k],rtol=1e-10,atol=1e-8))))
pd.DataFrame(results).to_csv(OUT/'tables/synthetic_saved_verification.csv',index=False)
assert all(x['passed'] for x in results)
(OUT/'metrics/synthetic_replay_status.json').write_text(json.dumps({'status':'passed','checks':len(results)},indent=2))
print(json.dumps({'pcapng_checks':checks,'synthetic_checks':len(results),'synthetic_passed':sum(x['passed'] for x in results)},indent=2))
