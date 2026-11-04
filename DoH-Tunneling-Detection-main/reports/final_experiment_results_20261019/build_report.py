"""Recalculate all report metrics from saved scores; no fitting or tuning on test."""
import json, math, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation import summarize
plt.rcParams.update({'font.size':11,'figure.dpi':130,'savefig.dpi':300,'svg.fonttype':'none'})
CHECKS=[]
def check(name,passed):
    CHECKS.append({'check':name,'passed':bool(passed)})
    assert passed,name
def md(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return 'NA' if not np.isfinite(x) else f'{x:.6g}'
        return str(x)
    return '| '+' | '.join(map(str,df.columns))+' |\n|'+'---|'*len(df.columns)+'\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'
def table(name,rows):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT/f'tables/{name}.csv',index=False)
    (OUT/f'tables/{name}.md').write_text(md(df),encoding='utf-8');return df
def save(name):
    plt.tight_layout();plt.savefig(OUT/f'figures/{name}.png');plt.savefig(OUT/f'figures/{name}.svg');plt.close()
def ci(k,n):
    return (0. if k==0 else beta.ppf(.025,k,n-k+1),1. if k==n else beta.ppf(.975,k+1,n-k),1. if k==n else beta.ppf(.95,k+1,n-k)) if n else (np.nan,)*3
def metrics(y,a):
    tp=int(sum((y==1)&a));fp=int(sum((y==0)&a));tn=int(sum((y==0)&~a));fn=int(sum((y==1)&~a))
    div=lambda x,n:x/n if n else np.nan
    r=dict(TP=tp,FP=fp,TN=tn,FN=fn,accuracy=div(tp+tn,len(y)),precision=div(tp,tp+fp),recall=div(tp,tp+fn),f1=div(2*tp,2*tp+fp+fn),fpr=div(fp,fp+tn),fnr=div(fn,tp+fn),n_benign=fp+tn,n_malicious=tp+fn,eligible=len(y),fpr_numerator=fp,fpr_denominator=fp+tn,fnr_numerator=fn,fnr_denominator=tp+fn)
    for name,k,n in [('fpr',fp,fp+tn),('fnr',fn,tp+fn),('recall',tp,tp+fn),('precision',tp,tp+fp),('accuracy',tp+tn,len(y))]:
        lo,hi,up=ci(k,n);r.update({name+'_ci95_low':lo,name+'_ci95_high':hi,name+'_upper95_one_sided':up})
    return r

def main():
    meta=pd.read_csv(OUT/'processed/flow_metadata.csv');split=pd.read_csv(OUT/'predictions/split_assignment.csv')
    capture=pd.read_csv(OUT/'metrics/capture_manifest.csv');elig=pd.read_csv(OUT/'tables/checkpoint_eligibility.csv')
    table('checkpoint_eligibility',elig);table('capture_inventory',capture)
    check('unique_flow_ids',meta.flow_id.is_unique)
    check('finite_monotonic_timestamps',meta.finite_timestamps.all() and meta.monotonic_timestamps.all())
    check('split_ids_unique',split.flow_id.is_unique)
    check('labels_match',split.set_index('flow_id').label.equals(meta.set_index('flow_id').loc[split.flow_id,'label']))
    split_counts=table('split_class_counts',split.groupby(['partition','label']).size().rename('flows').reset_index())
    table('capture_partition_counts',split.groupby(['source_file','partition','label']).size().rename('flows').reset_index())
    overlaps={}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        overlaps[a+'_'+b]=sorted(set(split[split.partition==a].source_file)&set(split[split.partition==b].source_file))
        check(a+'_'+b+'_flow_disjoint',not(set(split[split.partition==a].flow_id)&set(split[split.partition==b].flow_id)))
    (OUT/'metrics/capture_overlap.json').write_text(json.dumps(overlaps,indent=2))
    testids=set(split[split.partition=='test'].flow_id);testmeta=meta[meta.flow_id.isin(testids)];N=len(testmeta)
    common=set(testmeta[testmeta.stored_packets>=20].flow_id)
    allrows=[];costs=[];stops=[];errors=[];capture_rows=[];paired=[];rank=[];score_cache={};trace_cache={}
    for model in ['rf','cnn']:
        thresholds=json.loads((OUT/f'metrics/{model}_thresholds.json').read_text())
        df=pd.read_csv(OUT/f'predictions/{model}_test_prefixes.csv');val=pd.read_csv(OUT/f'predictions/{model}_val_prefixes.csv')
        check(model+'_finite_probabilities',np.isfinite(df.probability).all() and df.probability.between(0,1).all())
        check(model+'_score_labels',np.array_equal(df.label,meta.set_index('flow_id').loc[df.flow_id,'label']))
        check(model+'_test_membership',set(df.flow_id)==testids)
        score_cache[model]=df
        for policy in ['adaptive','fixed_10','fixed_20']:
            tau=thresholds[policy]['tau'];cols=[10,20] if policy=='adaptive' else [int(policy.split('_')[1])]
            for part,data in [('val',val),('test',df)]:
                pivot=data.pivot(index='flow_id',columns='checkpoint',values='probability').dropna(subset=[10,20])
                p=pivot[cols].to_numpy();hits=p>=tau;alert=hits.any(axis=1);stop=np.array(cols)[np.where(alert,hits.argmax(axis=1),len(cols)-1)]
                trace=pd.read_csv(OUT/f'predictions/{model}_{part}_{policy}_trace.csv').set_index('flow_id').loc[pivot.index].reset_index()
                check(model+part+policy+'_alerts',np.array_equal(alert,trace.alert))
                check(model+part+policy+'_observations',np.array_equal(stop,trace.observed_packets))
                y=trace.label.to_numpy();r=metrics(y,alert)
                prod=summarize(trace,20)
                for key in ['TP','FP','TN','FN','precision','recall','f1','fpr']:
                    check(model+part+policy+'_'+key,np.isclose(r[key],prod[key],equal_nan=True))
                if part=='val':
                    check(model+policy+'_val_fpr',r['fpr']<=.01)
                    sweep=pd.read_csv(OUT/f'metrics/{model}_{policy}_threshold_sweep.csv')
                    check(model+policy+'_lowest_feasible',np.isclose(sweep[sweep.fpr<=.01].iloc[0].tau,tau))
                    # Independent vectorized count for every saved validation grid threshold.
                    vy=meta.set_index('flow_id').loc[pivot.index,'label'].to_numpy();mx=p.max(axis=1)
                    fp=np.array([sum(mx[vy==0]>=t) for t in sweep.tau])
                    check(model+policy+'_sweep_fp',np.array_equal(fp,sweep.fp))
                    continue
                check(model+policy+'_common_population',set(trace.flow_id)==common)
                trace_cache[model,policy]=trace
                allrows.append(dict(model=model,policy=policy,checkpoint=20 if policy=='adaptive' else cols[0],population='common_ge20',tau=tau,excluded=N-len(y),outer_test_population=N,**r))
                det=(y==1)&alert;lat=trace.loc[det,'latency_s'];obs=trace.observed_packets
                costs.append(dict(model=model,policy=policy,n=len(y),mean_packets=obs.mean(),median_packets=obs.median(),p90_packets=obs.quantile(.9),p95_packets=obs.quantile(.95),cost_fraction_of_fixed20=obs.mean()/20,reduction_pct=100*(1-obs.mean()/20),malicious_mean_packets=obs[y==1].mean(),benign_mean_packets=obs[y==0].mean(),reaching_20_pct=100*(obs==20).mean(),detected_malicious=len(lat),mean_latency_s=lat.mean(),median_latency_s=lat.median(),p95_latency_s=lat.quantile(.95)))
                for c in [10,20]:
                    stops.append(dict(model=model,policy=policy,checkpoint=c,flows=int((obs==c).sum()),denominator=len(obs),percent=100*(obs==c).mean(),alerts=int(((obs==c)&alert).sum()),unresolved=int(((obs==c)&~alert).sum())))
                for source,g in trace.merge(meta[['flow_id','source_file']],on='flow_id').groupby('source_file'):
                    capture_rows.append(dict(model=model,policy=policy,capture=source,**metrics(g.label.to_numpy(),g.alert.to_numpy())))
            if policy!='adaptive':
                t=cols[0];native=df[df.checkpoint==t];a=native.probability.to_numpy()>=tau
                allrows.append(dict(model=model,policy=policy,checkpoint=t,population='native_prefix',tau=tau,excluded=N-len(native),outer_test_population=N,**metrics(native.label.to_numpy(),a)))
        a=trace_cache[model,'adaptive'].set_index('flow_id');f=trace_cache[model,'fixed_20'].set_index('flow_id').loc[a.index]
        scores=df.pivot(index='flow_id',columns='checkpoint',values='probability').loc[a.index]
        same=scores[20]>=thresholds['adaptive']['tau'];y=a.label
        errors.append(dict(model=model,n=len(a),extra_false_positives_vs_tuned_fixed20=int(((y==0)&a.alert&~f.alert).sum()),extra_false_negatives_vs_tuned_fixed20=int(((y==1)&~a.alert&f.alert).sum()),recovered_true_positives_vs_tuned_fixed20=int(((y==1)&a.alert&~f.alert).sum()),early_false_positives_vs_same_threshold_terminal=int(((y==0)&a.alert&~same).sum()),recovered_true_positives_vs_same_threshold_terminal=int(((y==1)&a.alert&~same).sum())))
        matched=(y==1)&a.alert&f.alert
        paired.append(dict(model=model,matched_true_positives=int(matched.sum()),adaptive_mean_latency_s=a.loc[matched,'latency_s'].mean(),fixed20_mean_latency_s=f.loc[matched,'latency_s'].mean(),mean_paired_saving_s=(f.loc[matched,'latency_s']-a.loc[matched,'latency_s']).mean()))
        for c in [10,20]:
            g=df[(df.checkpoint==c)&df.flow_id.isin(common)];y=g.label;prob=g.probability
            rank.append(dict(model=model,checkpoint=c,n=len(g),roc_auc=roc_auc_score(y,prob),average_precision=average_precision_score(y,prob)))
    perf=table('main_performance',allrows);cost=table('observation_cost_latency',costs);stopdf=table('stopping_distribution',stops)
    table('early_stopping_errors',errors);table('matched_detection_latency',paired);table('within_capture_performance',capture_rows);table('ranking_metrics',rank)
    short=table('main_performance_compact',perf[perf.population=='common_ge20'][['model','policy','TP','FP','TN','FN','precision','recall','f1','fpr','fpr_ci95_high','eligible','excluded']])
    # Plots deliberately read their CSV inputs, making saved tables the figure source.
    p=pd.read_csv(OUT/'tables/main_performance.csv');c=pd.read_csv(OUT/'tables/observation_cost_latency.csv');s=pd.read_csv(OUT/'tables/stopping_distribution.csv')
    fixed=p[(p.population=='common_ge20')&(p.policy!='adaptive')]
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,metric in zip(axes,['precision','recall','f1']):
        for model,g in fixed.groupby('model'):ax.plot(g.checkpoint,g[metric],'o-',label=model.upper())
        ax.set(xlabel='Packets observed',ylabel=metric.capitalize(),ylim=(-.03,1.05),xticks=[10,20]);ax.legend()
    fig.suptitle('Fixed-window performance: common ≥20-packet test cohort');save('precision_recall_f1')
    plt.figure(figsize=(7,4))
    for i,(model,g) in enumerate(fixed.groupby('model')):
        plt.errorbar(g.checkpoint+(i-.5)*.25,100*g.fpr,yerr=[100*(g.fpr-g.fpr_ci95_low),100*(g.fpr_ci95_high-g.fpr)],fmt='o-',capsize=4,label=model.upper())
    plt.axhline(1,color='black',linestyle='--',label='1% target');plt.xticks([10,20]);plt.xlabel('Packet prefix');plt.ylabel('FPR (%) — 95% exact binomial CI');plt.title('FPR uncertainty; flow independence is an optimistic assumption');plt.legend();save('fpr_confidence_intervals')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    labels=c.model.str.upper()+' '+c.policy
    axes[0].bar(labels,c.mean_packets,color=['#306998','#6baed6','#bdd7e7']*2);axes[0].set(ylabel='Mean packets observed',title='Actual observation cost')
    axes[1].bar(labels,c.mean_latency_s,color='#457b9d');axes[1].set(ylabel='Seconds since first packet',title='Latency among detected malicious flows')
    for ax in axes:ax.tick_params(axis='x',rotation=45)
    fig.suptitle('Common test cohort; latency excludes undetected attacks');save('observation_cost_and_latency')
    plt.figure(figsize=(7,4))
    for model in ['rf','cnn']:
        g=s[(s.model==model)&(s.policy=='adaptive')];plt.plot(g.checkpoint,g.percent,'o-',label=model.upper())
    plt.xticks([10,20]);plt.xlabel('Stopping checkpoint (packets)');plt.ylabel('Test flows (%)');plt.title('Adaptive stopping distribution, including terminal non-alerts');plt.legend();save('stopping_distribution')
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    for ax,r in zip(axes.flat,p[p.population=='common_ge20'].to_dict('records')):
        mat=np.array([[r['TN'],r['FP']],[r['FN'],r['TP']]]);ax.imshow(mat,cmap='Blues')
        for (i,j),v in np.ndenumerate(mat):ax.text(j,i,str(int(v)),ha='center',va='center',color='white' if v>mat.max()/2 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Benign','Attack'],yticklabels=['Benign','Attack'],xlabel='Predicted',ylabel='Actual',title=r['model'].upper()+' '+r['policy'])
    fig.suptitle('Confusion matrices: common ≥20-packet test cohort');save('confusion_matrices')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    curve_rows=[]
    for model in ['rf','cnn']:
        data=pd.read_csv(OUT/f'predictions/{model}_test_prefixes.csv')
        for cp in [10,20]:
            g=data[(data.checkpoint==cp)&data.flow_id.isin(common)];y=g.label;prob=g.probability
            x,t,_=roc_curve(y,prob);pr,re,_=precision_recall_curve(y,prob)
            axes[0].plot(x,t,label=f'{model.upper()} {cp} packets');axes[1].plot(re,pr,label=f'{model.upper()} {cp} packets')
            curve_rows.extend(dict(model=model,checkpoint=cp,curve='ROC',x=a,y=b) for a,b in zip(x,t));curve_rows.extend(dict(model=model,checkpoint=cp,curve='PR',x=a,y=b) for a,b in zip(re,pr))
    table('curve_coordinates',curve_rows)
    axes[0].plot([0,1],[0,1],'k--',alpha=.4);axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC from calibrated scores')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall from calibrated scores')
    for ax in axes:ax.legend(fontsize=8);ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02)
    save('roc_precision_recall')
    # Dataset selection graphic makes the principal coverage limitation visible.
    plt.figure(figsize=(7,4))
    for lab,g in elig[elig.capture=='ALL'].groupby('label'):plt.plot(g.checkpoint,100*g.eligible/g.total,'o-',label='Malicious' if lab else 'Benign')
    plt.xlabel('Required packet prefix');plt.ylabel('Eligible flows (%)');plt.title('Coverage of the complete extracted capture population');plt.xticks([10,20,30,40]);plt.legend();save('checkpoint_coverage')
    roles=pd.read_csv(OUT/'predictions/cnn_training_roles.csv')
    table('cnn_training_role_counts',roles.groupby(['role','label']).size().rename('flows').reset_index())
    check('cnn_roles_only_training',set(roles.flow_id)==set(split[split.partition=='train'].flow_id))
    check('cnn_roles_disjoint',roles.flow_id.is_unique)
    structures=json.loads((OUT/'metrics/pcapng_structure.json').read_text());check('pcapng_timing_formats_supported',all(r['latency_structure_supported'] for r in structures))
    mainp=perf[perf.population=='common_ge20'];ad=mainp[mainp.policy=='adaptive']
    text=['# Research results — preliminary within-capture evaluation','',
      '## 1. Experimental setup',
      'Fresh real-capture RF and CNN-GRU experiments, seed 42, packet checkpoints 10 and 20, validation target FPR 1%. Existing model architecture and first-crossing policy are retained. These are exploratory flow-isolated results, not an unseen-capture confirmation.',
      '## 2. Dataset characteristics and class distribution',
      f'Full parsing of seven unique captures produced {len(meta):,} TCP/443 flows: {(meta.label==0).sum():,} benign and {(meta.label==1).sum():,} malicious. Labels are filename/scenario labels (0 benign; 1 malicious), not independently verified application labels. Four benign and three malicious captures were hashed and processed without a packet limit.',
      md(elig[elig.capture=='ALL'].drop(columns='capture')),
      'Only 15.0053% of malicious flows reach 20 packets. No malicious flow reaches 30 or 40; those checkpoints cannot support a two-class detector comparison. The 10/20 schedule is already documented in the earlier audit and is a supported subset of the default checkpoints. It was selected for data support before examining new model scores.',
      '## 3. Train/validation/test splitting procedure',
      'The existing five-fold stratified group splitter rejects capture isolation with four benign and three malicious files. We explicitly use preliminary flow grouping: fold 0 test, fold 1 validation, remaining folds train. All prefixes stay with their flow. All seven captures occur across partitions; this is a major dependence/confounding limitation.',
      md(split_counts),
      f'The outer test has {N:,} flows reaching at least 10 packets. The common policy cohort has {len(common):,} flows reaching 20, with {N-len(common):,} excluded. Flows below 10 are excluded before splitting and documented in the dataset table. Native-prefix results in main_performance.csv separately retain all eligible 10-packet test flows.',
      '## 4. Models and stopping policies evaluated',
      'RF: 200 trees, minimum leaf size 3, balanced_subsample, equal total training weight per flow, five-fold flow-grouped out-of-fold Platt calibration. CNN-GRU: existing 64-channel two-convolution / two-layer 64-hidden GRU, up to 30 epochs with patience 7. Outer training is divided into fit, calibration, and early-stopping roles; threshold validation and final test remain separate. CNN weights prefixes equally; RF weights flows equally. The existing CNN minimum-improvement tolerance is 1e-6, so the saved epoch need not be the numerical minimum of every later validation loss.',
      'Adaptive alerts at the first probability ≥ its validation-selected threshold; otherwise it observes 20 packets and records no alert. Fixed 10 and fixed 20 have their own validation-selected thresholds. Each policy uses the lowest threshold on the existing 0.001 grid satisfying empirical validation FPR ≤1%. Validation threshold tuning uses the common ≥20 cohort; native-prefix test rows reuse those thresholds.',
      '## 5. Main performance comparison',md(short),
      'Full tables include accuracy, precision, recall, F1, FPR, FNR, exclusion counts, exact confidence bounds, and explicit event numerators and denominators. Missing predictions are not imputed as benign.',
      '## 6. Adaptive stopping and observation-cost findings',md(cost[['model','policy','mean_packets','median_packets','p95_packets','reduction_pct','reaching_20_pct','detected_malicious','mean_latency_s']]),md(pd.DataFrame(errors)),md(pd.DataFrame(paired)),
      'Observation reductions are calculated from actual stopping traces relative to fixed 20. Fixed 10 already costs exactly 10 packets, so reducing cost relative to fixed 20 does not by itself establish an advantage over the earliest fixed baseline. Paired latency uses only attacks detected by both adaptive and fixed 20. Ordinary latency summaries condition on detection and can be selection-biased when recall differs. RF adaptive and fixed 20 have equal aggregate error counts but flag different benign flows: adaptive adds three false positives relative to fixed 20 while avoiding three of fixed 20’s false positives. Its three additional early false positives also disappear when scoring only the terminal checkpoint at the adaptive threshold. CNN has no such decision disagreements.',
      '## 7. Statistical uncertainty and FPR assessment',
      '95% two-sided Clopper–Pearson exact binomial intervals and one-sided 95% FPR upper bounds are reported. These intervals assume independent flows and are optimistic under shared captures; they do not establish unseen-session error bounds. No FPR <1% claim follows merely from zero observed false positives.',
      md(ad[['model','FP','n_benign','fpr','fpr_ci95_low','fpr_ci95_high','fpr_upper95_one_sided']]),
      'At least 299 independent benign observations with zero false positives are required for a one-sided exact 95% upper bound below 1%; additional correlated flows are not a substitute for independent captures.',
      '## 8. Evasive-traffic findings',
      'No supplied per-flow rate, attack configuration, campaign or slow-tunneling labels identify genuine evasive groups. Prefix-length analysis is supported; an evasion-resistance conclusion is not. Existing timing-dilation code is an offline simulation, not measured slower tunneling. No synthetic timings are presented as real evasion findings. Needed evidence: labeled rate-controlled attack sessions, matched benign sessions, attack parameters, packet timestamps and capture-isolated partitions.',
      '## 9. Interpretation of the results']
    for model in ['rf','cnn']:
        g=mainp[mainp.model==model].set_index('policy');a=g.loc['adaptive'];f=g.loc['fixed_10'];cc=cost[(cost.model==model)&(cost.policy=='adaptive')].iloc[0]
        text.append(f"{model.upper()}: adaptive recall {a.recall:.4%}, FPR {a.fpr:.4%} ({int(a.FP)}/{int(a.n_benign)}), mean observation {cc.mean_packets:.4f} packets ({cc.reduction_pct:.2f}% reduction versus 20). Fixed 10 recall is {f.recall:.4%} and FPR {f.fpr:.4%}. These comparisons are conditional on the same long-flow cohort.")
    text += ['For both models, adaptive and fixed 10 produce the same classification counts and the same mean attack-detection latency (5.3965 ms). Adaptive observes about 10.62 packets across all flows versus exactly 10 for fixed 10; it therefore shows no measured benefit over fixed 10 here. Against fixed 20, mean detected-attack latency drops from 6.2466 ms to 5.3965 ms, a paired saving of 0.8501 ms. CNN zero false positives among 275 benign flows still yields a one-sided 95% FPR upper bound of 1.0834% (two-sided upper endpoint 1.3325%). RF empirical FPR is 1.0909%, exceeding the target.',
      '## 10. Threats to validity and limitations',
      'Shared capture sessions, extreme class imbalance, filename-derived labels, selection against short attacks, and possible handshake/host/tool fingerprints prevent generalization claims. TCP/443 filtering alone does not authenticate DoH. One seed and one flow split do not estimate split variability. Confidence intervals do not account for capture correlation. Parser retransmission filtering uses a sequence high-water mark and can drop unseen reordered payload; timestamps are sorted only after extraction and only the first 200 retained packets are stored. See preprocessing counters. Complete structural inspection found no mixed-endian sections, non-Ethernet interfaces, simple packet blocks or nonzero timestamp offsets in these PCAPNG files; this narrows but does not eliminate parser limitations.',
      'The earlier audit describes two real-data runs from a different workspace. Their score files, processed data and model artifacts are absent here; their numerical findings were not copied into this study. Saved synthetic scores were independently checked only as software evidence. The existing 113-test suite passed. An initial RF launch preceded completion of split preparation and failed without producing model results; it was rerun successfully after preparation. Logs preserve both attempts.',
      '## 11. Overall conclusion',
      'These results measure the adaptive policy’s cost/error trade-off on a common long-flow subset of real captures. Adaptive stopping reduces observation versus fixed 20 but provides no measured benefit over fixed 10. They do not establish that unseen-capture FPR is approximately 1% or that the detector resists evasive traffic. The common-cohort measurements and their exclusions are suitable for an explicitly preliminary report; a confirmatory generalization claim requires additional independent capture sessions and a prespecified short-flow protocol.']
    confpath=OUT/'metrics/confounding_diagnostics.json'
    if confpath.exists():
        conf=json.loads(confpath.read_text())
        text.insert(-2,f"Additional dependence diagnostic: {conf['common_prefix_signatures_shared_train_test']} distinct first-20 length/direction signatures occur in both training and test. Across the full common-horizon population, {conf['flows_in_repeated_common_prefix_signatures']:,} flows belong to repeated signature groups. These signatures omit timing and do not prove duplicate capture bytes, but reinforce the need for capture-isolated evaluation.")
    (OUT/'results_summary.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    (OUT/'methodology_and_limitations.md').write_text('''# Methodology and limitations

Read results_summary.md for the full protocol and numerical findings. This package contains fresh real-capture measurements, independently recomputed saved synthetic software checks, and explicit documentation of unavailable historical real runs. No synthetic result is a real-data finding.

The unit of prediction is a bidirectional TCP/443 flow, separated on a 120-second idle gap or SYN after closure. ACK/SYN/FIN packets are included. Length is IP total length, direction is relative to first packet observed. Retransmissions use the repository high-water heuristic. Labels: filename Benign → 0; Exfiltration → 1. Unique hashes establish byte identity, not independent session provenance or protocol correctness.

Confusion definitions: TP = alert on malicious; FP = alert on benign; TN = no alert on benign; FN = no alert on malicious. Accuracy=(TP+TN)/N, precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN), FPR=FP/(FP+TN), FNR=FN/(TP+FN). Undefined metrics use NA. Rates in CSV are fractions; explicitly named percent columns are percentages. Two-sided 95% exact binomial intervals use beta quantiles 0.025 and 0.975; one-sided 95% upper bounds use 0.95. They are conditional flow-level intervals, not cluster-adjusted bounds.

Observation cost = retained packets through decision, including the alert packet; non-alerts reach the terminal checkpoint. Cost fraction divides mean packets by 20. Reduction=100*(1-cost). Percentiles use pandas linear interpolation. Latency is capture time at alert packet minus capture time of first retained packet, in seconds; it excludes inference/processing delay and is only defined for detected malicious flows in performance summaries. It is not time since attack onset. Paired detection latency conditions on true positives under both policies.

The native 10-packet cohort differs from the common ≥20-packet cohort. Fixed native-prefix rows reuse the threshold tuned on common validation flows and therefore do not claim an independently tuned native-cohort FPR constraint. Flow-level exclusions count outer test flows shorter than the evaluated horizon; globally excluded <10 flows are separate in checkpoint_eligibility.csv. All 30/40-packet malicious counts are zero; no performance plot at these checkpoints is defensible.

Early stopping errors are compared both with independently tuned fixed 20 and with terminal scoring at the same adaptive threshold. The latter isolates additional first-crossing alerts. At an identical threshold the adaptive policy cannot lose terminal alerts. Capture-specific tables are descriptive within-capture results, not held-out capture generalization. No capture-generalization figure was produced. No evasion figure was produced because real rate/attack group labels are absent.

The existing splitter requires at least five captures per class: one additional benign and two additional malicious captures are the minimum merely to pass that outer-split guard. More independent groups may be required to populate all CNN fit, early-stop, calibration, validation and test roles with both classes. A passing group count is not proof of statistical independence. Obtain host/tool/session provenance, substantial independent benign test coverage, rate-controlled attacks and a prespecified policy for flows ending before the horizon. Evaluate capture-level uncertainty using enough genuinely independent sessions.

Source is frozen in source_snapshot/. Existing production source, original captures and old results were not modified. The experiment harness adds model/calibrator persistence and a separate CNN early-stopping role. CPU threads=4; seeds are 42 (outer, calibration, training) and 43 (inner early-stop split). Architecture, RF defaults and stopping grid are unchanged. CNN selection retains the existing 1e-6 improvement tolerance. Historical dataset publication/date/tool claims in the README have not been independently authenticated and are not used as evidence here.
''',encoding='utf-8')
    (OUT/'reproduction.md').write_text('''# Reproduction

From `D:\\DoH-Tunneling-Detection-main`, using the existing environment:

```powershell
.\\.venv\\Scripts\\python.exe reports/final_experiment_results_20261019/build_report.py
.\\.venv\\Scripts\\python.exe reports/final_experiment_results_20261019/verify_models.py
```

The first command regenerates metrics and figures from saved real predictions; it does not train. The second checks fitted bundle predictions and timestamps. These regenerate derived reports/check records; preserve a copy if editing the reporting scripts.

To rerun full extraction/training without overwriting this package, create a NEW directory under reports, copy run_experiments.py, verify_inputs.py, build_report.py and verify_models.py into it, and copy source_snapshot/config.py and source_snapshot/src into the root of an isolated workspace with the same seven captures under data. Install the versions recorded in metrics/environment.json. Then, replacing RESULT_DIR with the new relative path, run SEQUENTIALLY:

```powershell
.\\.venv\\Scripts\\python.exe -u RESULT_DIR/run_experiments.py prepare
.\\.venv\\Scripts\\python.exe RESULT_DIR/verify_inputs.py
.\\.venv\\Scripts\\python.exe -u RESULT_DIR/run_experiments.py rf
.\\.venv\\Scripts\\python.exe -u RESULT_DIR/run_experiments.py cnn
.\\.venv\\Scripts\\python.exe RESULT_DIR/build_report.py
.\\.venv\\Scripts\\python.exe RESULT_DIR/verify_models.py
```

For synthetic saved-evidence checks, also retain data/results/synthetic_smoke_test. No rerun of synthetic training is required. Model bundles are local trusted Python pickles; source and library versions matter. Model scores, raw probabilities, calibration maps, full policy threshold sweeps, per-flow stopping traces, capture hashes and split roles are saved. CPU floating-point/training behavior across versions can differ; reproduce the recorded environment for closest agreement.

Software validation used:
```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests -q -p no:cacheprovider --basetemp reports/final_experiment_results_20261019/test_temp
```
113 tests passed; this validates software fixtures, not unseen-capture performance. The initial rf.log records a premature launch before split_assignment.csv existed; rf_run.log records the completed retry.
''',encoding='utf-8')
    (OUT/'metrics/report_verification.json').write_text(json.dumps({'checks':CHECKS,'passed':sum(x['passed'] for x in CHECKS),'failed':sum(not x['passed'] for x in CHECKS)},indent=2))
    extra='''
Exact commands to create a fresh run using the unchanged source in this workspace:

```powershell
Set-Location 'D:\\DoH-Tunneling-Detection-main'
$resultDir = 'reports/final_experiment_results_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $resultDir -ErrorAction Stop
Copy-Item -Path 'reports/final_experiment_results_20261019/*.py' -Destination $resultDir
& '.\\.venv\\Scripts\\python.exe' -u "$resultDir/run_experiments.py" prepare
& '.\\.venv\\Scripts\\python.exe' "$resultDir/verify_inputs.py"
& '.\\.venv\\Scripts\\python.exe' -u "$resultDir/run_experiments.py" rf
& '.\\.venv\\Scripts\\python.exe' -u "$resultDir/run_experiments.py" cnn
& '.\\.venv\\Scripts\\python.exe' "$resultDir/build_report.py"
& '.\\.venv\\Scripts\\python.exe' "$resultDir/verify_models.py"
```

Check each command exits successfully before continuing. Existing model bundles and processed datasets are guarded against overwrite.
'''
    with (OUT/'reproduction.md').open('a',encoding='utf-8') as f:f.write(extra)
    figures=['checkpoint_coverage','precision_recall_f1','fpr_confidence_intervals','observation_cost_and_latency','confusion_matrices','roc_precision_recall','stopping_distribution']
    (OUT/'figures/index.md').write_text('# Report figures\n\nAll figures use real-capture preliminary within-capture outputs. Each PNG is 300 dpi; each SVG keeps editable text.\n\n'+'\n\n'.join(f'## {name.replace("_"," ").title()}\n\n![{name}]({name}.png)\n\n[Editable SVG]({name}.svg)' for name in figures),encoding='utf-8')
    for name in ['synthetic_saved_verification','first_packet_length_distribution']:
        path=OUT/f'tables/{name}.csv'
        if path.exists():table(name,pd.read_csv(path))
    print(short.to_string(index=False));print('Report checks:',len(CHECKS),'passed')

if __name__=='__main__':main()
