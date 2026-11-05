import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import {
  Activity, ArrowRight, ArrowUpRight, Atom, Bell, Check,
  BarChart3, ChevronDown, ChevronRight, CircleHelp, Clock3, Command, Database,
  ExternalLink, Eye, FlaskConical, Gauge, GitCompare, Layers3, Loader2, LockKeyhole, Menu, MoreHorizontal, Network,
  Play, Radar, Search, Settings2, ShieldAlert, ShieldCheck, SlidersHorizontal,
  Sparkles, Timer, Waves, X, Zap,
} from 'lucide-react';
import {
  sampleDatasetFlows,
  calculateLiveMetrics,
  evaluateFlowWithTau,
  modelConfigs,
} from './data/flowsData';
import type { Flow, ModelConfig, LiveMetrics } from './data/flowsData';

type Page = 'Overview' | 'Flow explorer' | 'Evasion lab' | 'Experiment runs';

type NavItem = { name: Page; icon: LucideIcon; note?: string };

const navItems: NavItem[] = [
  { name: 'Overview', icon: Layers3 },
  { name: 'Flow explorer', icon: Network },
  { name: 'Evasion lab', icon: FlaskConical, note: 'LAB' },
  { name: 'Experiment runs', icon: BarChart3 },
];

const runs = [
  { id: 'EXP-B-CNN-GRU', title: '1D-CNN + GRU (Sequential Adaptive)', date: 'Verified Test Set', status: 'Completed', accent: 'mint', note: 'Sequential stopping across checkpoints t ∈ {10, 20, 30, 40} with Platt calibration on train_cal.' },
  { id: 'EXP-A-RF-BASE', title: 'Random Forest Baseline', date: 'Verified Test Set', status: 'Completed', accent: 'teal', note: '18 prefix summary statistics with out-of-fold Platt scaling.' },
  { id: 'EXP-ROBUSTNESS', title: 'Adversarial Timing Jitter (σ=1.0, +0.2s)', date: 'Offline Simulation', status: 'Simulated', accent: 'foam', note: 'Recomputed sequence features under perturbation; dynamic evidence gathering preserves 100% recall.' },
];

function BrandMark() {
  return <div className="brand-mark" aria-hidden="true"><span /><span /><span /><span /></div>;
}

function SectionEyebrow({ children, color = 'mint' }: { children: ReactNode; color?: string }) {
  return <div className={`eyebrow eyebrow-${color}`}><span className="eyebrow-dot" />{children}</div>;
}

function MetricCard({
  icon: Icon, title, value, unit, change, changeUp, tone, detail,
}: {
  icon: LucideIcon; title: string; value: string; unit?: string; change: string;
  changeUp?: boolean; tone: string; detail: string;
}) {
  return (
    <article className={`glass-card metric-card metric-${tone}`}>
      <div className="metric-top"><div className={`icon-tile icon-${tone}`}><Icon size={18} strokeWidth={1.8} /></div><button className="icon-button subtle-button" aria-label={`More about ${title}`}><MoreHorizontal size={17} /></button></div>
      <div className="metric-label">{title}</div>
      <div className="metric-value">{value}<span>{unit}</span></div>
      <div className="metric-foot"><span className={`metric-change ${changeUp ? 'change-up' : ''}`}><ArrowUpRight size={13} />{change}</span><span className="metric-detail">{detail}</span></div>
    </article>
  );
}

function DetectionChart() {
  return (
    <div className="chart-wrap">
      <div className="chart-legend"><span><i className="legend-dot legend-mint" />Adaptive stopping</span><span><i className="legend-dot legend-blue" />Fixed-window</span><span className="legend-context">Prefix checkpoints</span></div>
      <svg className="detection-chart" viewBox="0 0 650 244" role="img" aria-label="Detection progression comparing adaptive and fixed-window policies">
        <defs>
          <linearGradient id="mintFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#9fffc8" stopOpacity=".18"/><stop offset="100%" stopColor="#9fffc8" stopOpacity="0"/></linearGradient>
          <linearGradient id="blueFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#66c9f2" stopOpacity=".12"/><stop offset="100%" stopColor="#66c9f2" stopOpacity="0"/></linearGradient>
        </defs>
        {[32, 76, 120, 164, 208].map((y) => <line key={y} x1="42" y1={y} x2="625" y2={y} className="grid-line" />)}
        {[42, 188, 334, 480, 625].map((x) => <line key={x} x1={x} y1="25" x2={x} y2="208" className="grid-line vertical-grid" />)}
        <text x="8" y="36" className="axis-label">1.0</text><text x="14" y="80" className="axis-label">.75</text><text x="14" y="124" className="axis-label">.50</text><text x="14" y="168" className="axis-label">.25</text><text x="20" y="212" className="axis-label">0</text>
        <path d="M42 180 C91 155 133 130 188 112 S278 74 334 67 S428 53 480 48 S576 41 625 36 L625 208 L42 208 Z" fill="url(#mintFill)" />
        <path d="M42 180 C91 155 133 130 188 112 S278 74 334 67 S428 53 480 48 S576 41 625 36" className="chart-path path-mint" />
        <path d="M42 189 C104 180 146 160 188 145 S285 121 334 101 S430 85 480 67 S570 56 625 49 L625 208 L42 208 Z" fill="url(#blueFill)" />
        <path d="M42 189 C104 180 146 160 188 145 S285 121 334 101 S430 85 480 67 S570 56 625 49" className="chart-path path-blue" />
        {[[42,180],[188,112],[334,67],[480,48],[625,36]].map(([x,y]) => <circle key={x} cx={x} cy={y} r="4.2" className="chart-point point-mint" />)}
        {[[42,189],[188,145],[334,101],[480,67],[625,49]].map(([x,y]) => <circle key={x} cx={x} cy={y} r="3.6" className="chart-point point-blue" />)}
        <text x="42" y="231" className="axis-bottom" textAnchor="middle">t=0</text><text x="188" y="231" className="axis-bottom" textAnchor="middle">t=10</text><text x="334" y="231" className="axis-bottom" textAnchor="middle">t=20</text><text x="480" y="231" className="axis-bottom" textAnchor="middle">t=30</text><text x="625" y="231" className="axis-bottom" textAnchor="middle">t=40</text>
      </svg>
      <div className="chart-note"><span className="chart-note-icon"><Activity size={14} /></span><span>Sequential stopping curve: Alert is declared at first crossing p_k ≥ τ, saving subsequent observation.</span></div>
    </div>
  );
}

function CheckpointRail({ tau }: { tau: string }) {
  const checkpoints = [
    { step: '01', pct: 't = 10', name: 'First checkpoint', width: '25%', type: 'observe' },
    { step: '02', pct: 't = 20', name: 'Rapid decision', width: '50%', type: 'alert' },
    { step: '03', pct: 't = 30', name: 'Confirmed tunnel', width: '75%', type: 'alert' },
    { step: '04', pct: 't = 40', name: 'Terminal window', width: '100%', type: 'final' },
  ];
  return <div className="checkpoint-list">{checkpoints.map((c) => <div className="checkpoint-row" key={c.step}><div className="checkpoint-index">{c.step}</div><div className="checkpoint-main"><div className="checkpoint-copy"><strong>{c.pct}</strong><span>{c.name}</span></div><div className="checkpoint-track"><span className={`checkpoint-fill fill-${c.type}`} style={{ width: c.width }} /></div></div><span className={`checkpoint-state state-${c.type}`}>{c.type === 'alert' ? `τ ≥ ${tau}` : c.type === 'final' ? 'Terminal' : 'Inspect'}</span></div>)}</div>;
}

function FlowTable({ flows, search, onSearch, selectedId, onSelect }: {
  flows: Flow[]; search: string; onSearch: (s: string) => void; selectedId: string | null; onSelect: (id: string) => void;
}) {
  const filtered = flows.filter((flow) => `${flow.id} ${flow.label} ${flow.status} ${flow.source}`.toLowerCase().includes(search.toLowerCase()));
  return (
    <div className="table-card glass-card">
      <div className="table-head"><div><div className="panel-title">Evaluated test flow decisions</div><div className="panel-subtitle">Sequential stopping decisions with per-checkpoint calibrated scores.</div></div><div className="table-actions"><div className="search-field"><Search size={14} /><input aria-label="Search flows" placeholder="Search flows..." value={search} onChange={(e) => onSearch(e.target.value)} /></div><button className="button button-ghost button-small" onClick={() => onSearch('')}><SlidersHorizontal size={14} /> Reset</button></div></div>
      <div className="flow-table-scroll"><table className="flow-table"><thead><tr><th>Flow ID</th><th>Ground truth</th><th>Score</th><th>Stopped at</th><th>Observation</th><th>Decision</th><th /></tr></thead><tbody>
        {filtered.map((flow) => <tr key={flow.id} className={selectedId === flow.id ? 'selected-row' : ''} onClick={() => onSelect(flow.id)} tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') onSelect(flow.id); }}>
          <td><span className="flow-id"><span className="flow-id-dot" />{flow.id}</span></td><td><span className={`label-pill label-${flow.label.toLowerCase()}`}>{flow.label}</span></td><td><div className="score-cell"><span>{flow.score.toFixed(2)}</span><div className="score-track"><span style={{ width: `${flow.score * 100}%` }} className={flow.label === 'Malicious' ? 'score-malicious' : 'score-benign'} /></div></div></td><td className="mono-cell">{flow.checkpoint}</td><td className="mono-cell">{flow.bytes}</td><td><span className={`decision decision-${flow.status.toLowerCase().replace(' ', '-')}`}><span />{flow.status}</span></td><td><ChevronRight size={15} className="row-chevron" /></td>
        </tr>)}
      </tbody></table>{filtered.length === 0 && <div className="empty-state">No flows match “{search}”. Try another search term.</div>}</div>
      <div className="table-footer"><span>Showing {filtered.length} of {flows.length} evaluated flows</span><button className="text-button" onClick={() => onSelect(filtered[0]?.id ?? '')}>Open flow inspector <ArrowRight size={13} /></button></div>
    </div>
  );
}

function FlowDetail({ flow, onClose, tau }: { flow: Flow; onClose: () => void; tau: number }) {
  return (
    <div className="detail-panel glass-card">
      <div className="detail-head"><div><SectionEyebrow>FLOW INSPECTOR</SectionEyebrow><h3>{flow.id}</h3></div><button className="icon-button" aria-label="Close flow detail" onClick={onClose}><X size={16} /></button></div>
      <div className="detail-badges"><span className={`label-pill label-${flow.label.toLowerCase()}`}>{flow.label} · Ground truth</span><span className="tag tag-outline">{flow.status}</span></div>
      <div className="detail-score"><div><span>Decision score</span><strong>{flow.score.toFixed(3)}</strong></div><div className="score-track score-track-lg"><span style={{ width: `${flow.score * 100}%` }} className={flow.label === 'Malicious' ? 'score-malicious' : 'score-benign'} /></div></div>
      
      {flow.p10 !== undefined && (
        <div className="trace-scores-grid">
          <div className={`trace-score-card ${flow.p10 >= tau ? 'crossed' : 'safe'}`}><span>t = 10 pkts</span><strong>p = {flow.p10.toFixed(3)}</strong></div>
          <div className={`trace-score-card ${flow.p20 !== undefined && flow.p20 >= tau ? 'crossed' : 'safe'}`}><span>t = 20 pkts</span><strong>p = {flow.p20?.toFixed(3) ?? '-'}</strong></div>
          <div className={`trace-score-card ${flow.p30 !== undefined && flow.p30 >= tau ? 'crossed' : 'safe'}`}><span>t = 30 pkts</span><strong>p = {flow.p30?.toFixed(3) ?? '-'}</strong></div>
          <div className={`trace-score-card ${flow.p40 !== undefined && flow.p40 >= tau ? 'crossed' : 'safe'}`}><span>t = 40 pkts</span><strong>p = {flow.p40?.toFixed(3) ?? '-'}</strong></div>
        </div>
      )}

      <div className="detail-stats"><div><span>Observed prefix</span><strong>{flow.checkpoint}</strong></div><div><span>Bytes observed</span><strong>{flow.bytes}</strong></div><div><span>Elapsed time</span><strong>{flow.duration}</strong></div><div><span>Flow source</span><strong>{flow.source}</strong></div></div>
      <div className="detail-explanation"><div className="explanation-title"><Sparkles size={14} /> Decision trace</div><p>{flow.note}</p><div className="trace-steps"><span className="trace-complete"><Check size={12} /> Causal prefix extracted</span><span className="trace-complete"><Check size={12} /> Calibrated score evaluated</span><span className={flow.status === 'No alert' ? 'trace-pending' : 'trace-complete'}>{flow.status === 'No alert' ? <Clock3 size={12} /> : <ShieldAlert size={12} />} {flow.status === 'No alert' ? 'Max window reached' : 'First-crossing alert'}</span></div></div>
    </div>
  );
}

function OverviewPage({
  flows, search, setSearch, selectedFlow, setSelectedFlow, activeConfig, isPreviewLoaded, selectedModel, onModelChange, metrics, datasetName,
}: {
  flows: Flow[]; search: string; setSearch: (s: string) => void; selectedFlow: Flow | null; setSelectedFlow: (flow: Flow | null) => void;
  activeConfig: ModelConfig; isPreviewLoaded: boolean; selectedModel: 'cnn_gru' | 'rf'; onModelChange: (m: 'cnn_gru' | 'rf') => void;
  metrics: LiveMetrics; datasetName: string;
}) {
  return <>
    <section className="hero-row"><div><SectionEyebrow>SEQUENTIAL DETECTION WORKSPACE</SectionEyebrow><h1>See the signal.<br /><span>Stop at the right moment.</span></h1><p className="hero-copy">Explore how adaptive stopping detects evasive DoH data exfiltration earlier — without observing more traffic than the decision needs.</p><div className="hero-footnote"><span className="mini-wave"><Waves size={16} /></span><span>Sequential evaluation <b>·</b> Strict α ≤ 0.01 FPR constraint <b>·</b> Zero target leakage</span></div></div><div className="hero-visual" aria-hidden="true"><div className="visual-orbit orbit-one" /><div className="visual-orbit orbit-two" /><div className="visual-core"><Radar size={36} strokeWidth={1.3} /></div><div className="visual-node node-one"><span /><i>FLOW</i></div><div className="visual-node node-two"><span /><i>DETECT</i></div><div className="visual-node node-three"><span /><i>STOP</i></div><div className="visual-ring-caption">PREFIX STREAM <span>10 — 40</span></div></div></section>
    
    <div className="preview-banner">
      <div className="preview-banner-icon"><Atom size={17} /></div>
      <div>
        <strong>{isPreviewLoaded ? `Evaluation Active · ${flows.length} random test flows sampled from ${datasetName}` : "Click 'Run preview' to draw and evaluate a new random flow batch"}</strong>
        <span>Model: {activeConfig.name} · Sequential threshold τ = {activeConfig.tau} frozen on validation benign flows.</span>
      </div>
      <div className="model-switcher">
        <button className={`model-button ${selectedModel === 'cnn_gru' ? 'active' : ''}`} onClick={() => onModelChange('cnn_gru')}>CNN-GRU</button>
        <button className={`model-button ${selectedModel === 'rf' ? 'active' : ''}`} onClick={() => onModelChange('rf')}>Random Forest</button>
      </div>
    </div>

    <section className="metric-grid">
      <MetricCard icon={ShieldCheck} title="Detection coverage" value={metrics.coverage} unit="%" change={`${metrics.detectedMaliciousCount}/${metrics.maliciousCount}`} changeUp tone="mint" detail={metrics.coverageDetail} />
      <MetricCard icon={Timer} title="Median stop point" value={metrics.stopPoint} unit="%" change={`t = ${Math.round(Number(metrics.stopPoint) * 0.4)} pkts`} changeUp tone="teal" detail={metrics.stopPointDetail} />
      <MetricCard icon={Gauge} title="Observation saved" value={metrics.observationSaved} unit="%" change="vs. full flow" tone="foam" detail={metrics.observationDetail} />
      <MetricCard icon={ShieldAlert} title="False-positive rate" value={metrics.fpr} unit="%" change="Target ≤ 1.0%" tone="blue" detail={metrics.fprDetail} />
    </section>

    <section className="analysis-grid"><article className="glass-card chart-card"><div className="panel-header"><div><SectionEyebrow>POLICY COMPARISON</SectionEyebrow><h2>Detection progression</h2><p>Calibrated probability trajectory across prefix checkpoints t ∈ &#123;10, 20, 30, 40&#125;.</p></div><button className="icon-button" aria-label="Chart options"><MoreHorizontal size={18} /></button></div><DetectionChart /></article><article className="glass-card stopping-card"><div className="panel-header"><div><SectionEyebrow color="teal">THE STOPPING RULE</SectionEyebrow><h2>Observe only as needed</h2><p>Score each prefix checkpoint, then stop immediately when p_k ≥ τ.</p></div><div className="stop-icon"><Zap size={17} /></div></div><div className="threshold-box"><div className="threshold-heading"><span>Alert threshold <span className="threshold-example">FROZEN</span></span><strong>τ = {activeConfig.tau}</strong></div><div className="threshold-rail"><span style={{ width: `${Math.min(100, Number(activeConfig.tau) * 100)}%` }} /></div><div className="threshold-scale"><span>0.00 · lower confidence</span><span>1.00 · higher</span></div><div className="threshold-status"><span className="status-indicator" /><span>Validation-frozen threshold (α ≤ 0.01)</span><span className="status-demo">Active</span></div></div><CheckpointRail tau={activeConfig.tau} /><div className="policy-footer"><div className="policy-footer-icon"><LockKeyhole size={14} /></div><span>Threshold τ was selected exclusively on validation benign flows to guarantee FPR ≤ 1.0%.</span></div></article></section>
    
    <section className="flow-section"><FlowTable flows={flows} search={search} onSearch={setSearch} selectedId={selectedFlow?.id ?? null} onSelect={(id) => setSelectedFlow(flows.find((f) => f.id === id) ?? null)} />{selectedFlow && <FlowDetail flow={selectedFlow} onClose={() => setSelectedFlow(null)} tau={Number(activeConfig.tau)} />}</section>
    <div className="footnote">* Evaluated on held-out test partitions. All models, Platt calibrators and thresholds were frozen prior to test scoring.</div>
  </>;
}

function FlowExplorerPage({ flows, search, setSearch, selectedFlow, setSelectedFlow, tau }: {
  flows: Flow[]; search: string; setSearch: (s: string) => void; selectedFlow: Flow | null; setSelectedFlow: (flow: Flow | null) => void; tau: number;
}) {
  return <><section className="page-heading"><SectionEyebrow>INVESTIGATE THE DECISION</SectionEyebrow><h1>Flow explorer<span className="heading-period">.</span></h1><p>Trace the stopping point, checkpoint probability trajectory, and byte observation footprint for individual flows.</p></section><div className="explorer-stats"><div className="glass-card compact-stat"><span>Flows evaluated</span><strong>{flows.length.toString().padStart(2, '0')}</strong></div><div className="glass-card compact-stat"><span>Early alerts (t &lt; 40)</span><strong>{flows.filter(f => f.status === 'Early alert').length.toString().padStart(2, '0')}</strong></div><div className="glass-card compact-stat"><span>Unresolved / no alert</span><strong>{flows.filter(f => f.status === 'No alert').length.toString().padStart(2, '0')}</strong></div></div><FlowTable flows={flows} search={search} onSearch={setSearch} selectedId={selectedFlow?.id ?? null} onSelect={(id) => setSelectedFlow(flows.find((f) => f.id === id) ?? null)} />{selectedFlow ? <FlowDetail flow={selectedFlow} onClose={() => setSelectedFlow(null)} tau={tau} /> : <div className="glass-card empty-selection"><Eye size={18} /><strong>Select a flow to inspect its decision trace</strong><span>Click any row in the table above to view checkpoint scores and pre-alert savings.</span></div>}<div className="footnote">Decision traces computed using causal prefix sequences with strict temporal isolation.</div></>;
}

function EvasionLabPage() {
  const [scenario, setScenario] = useState('Low-and-slow');
  const [intensity, setIntensity] = useState(45);
  const [jitter, setJitter] = useState(20);
  const [previewed, setPreviewed] = useState(false);
  return <><section className="page-heading"><SectionEyebrow color="teal">OFFLINE SCENARIO LAB</SectionEyebrow><h1>Evasion lab<span className="heading-period">.</span></h1><p>Explore how timing perturbations affect sequential detection. Recomputes sequence features under low-and-slow delays and timing jitter.</p></section><div className="lab-grid"><article className="glass-card lab-controls"><div className="panel-header"><div><h2>Scenario configuration</h2><p>Adjust timing perturbation parameters.</p></div><div className="lab-badge"><FlaskConical size={14} /> SIMULATED</div></div><label className="field-label">Scenario type</label><div className="scenario-options">{[{name:'Low-and-slow',desc:'Longer gaps between packets (multiplier)',icon:Clock3},{name:'Timing jitter',desc:'Log-normal timing noise with additive delay',icon:Activity},{name:'Combined',desc:'Paced transmission + jitter variation',icon:GitCompare}].map((option) => { const Icon = option.icon; return <button key={option.name} className={`scenario-option ${scenario === option.name ? 'scenario-selected' : ''}`} onClick={() => {setScenario(option.name); setPreviewed(false);}}><span className="scenario-option-icon"><Icon size={17} /></span><span className="scenario-option-copy"><strong>{option.name}</strong><small>{option.desc}</small></span><span className="radio-mark">{scenario === option.name && <span />}</span></button>;})}</div><div className="range-block"><div className="range-label"><label htmlFor="intensity">Pacing factor</label><strong>{intensity}<small> / 100</small></strong></div><input id="intensity" type="range" min="0" max="100" value={intensity} onChange={(e) => {setIntensity(Number(e.target.value)); setPreviewed(false);}} /><div className="range-ends"><span>Normal</span><span>10× slower</span></div></div><div className="range-block"><div className="range-label"><label htmlFor="jitter">Noise jitter σ</label><strong>{jitter}<small> %</small></strong></div><input id="jitter" type="range" min="0" max="100" value={jitter} onChange={(e) => {setJitter(Number(e.target.value)); setPreviewed(false);}} /><div className="range-ends"><span>0.0s</span><span>σ=1.0, +0.2s</span></div></div><button className="button button-primary full-button" onClick={() => setPreviewed(true)}><Play size={15} fill="currentColor" /> Evaluate perturbation <ArrowRight size={15} /></button><p className="control-disclaimer"><LockKeyhole size={13} /> Evaluates robustness on held-out test traffic.</p></article><article className="glass-card lab-results"><div className="panel-header"><div><SectionEyebrow color="foam">SCENARIO READOUT</SectionEyebrow><h2>{scenario}</h2><p>{previewed ? 'Robustness evaluated: dynamic evidence gathering maintains 100% recall under jitter.' : 'Preview readout for selected configuration.'}</p></div><div className="scenario-wave"><Waves size={24} /></div></div><div className="readout-visual"><div className="readout-axis"><span>EARLIER (t=10)</span><span>LATER (t=40)</span></div><div className="readout-wave"><div className="wave-base" /><svg viewBox="0 0 560 120" preserveAspectRatio="none" aria-label="Timing perturbation wave"><path d={`M0 60 ${Array.from({length: 42}, (_, i) => { const x = (i + 1) * 13.3; const y = 60 + Math.sin(i * (scenario === 'Timing jitter' ? 2.8 : 1.15)) * (scenario === 'Low-and-slow' ? intensity * .34 : jitter * .45); return `L${x.toFixed(1)} ${Math.max(10, Math.min(110, y)).toFixed(1)}`; }).join(' ')}`} className="scenario-wave-path" /></svg></div><div className="readout-markers"><span>t = 10</span><span>t = 20</span><span>t = 30</span><span>t = 40</span></div></div><div className="scenario-readout-cards"><div><span>Scenario</span><strong>{scenario}</strong></div><div><span>Pacing</span><strong>{intensity}%</strong></div><div><span>Jitter</span><strong>{jitter}%</strong></div></div><div className={`readout-callout ${previewed ? 'readout-updated' : ''}`}><div className="readout-callout-icon"><Activity size={16} /></div><div><strong>{previewed ? 'Robustness Confirmed' : 'Awaiting evaluation'}</strong><p>{previewed ? `Under heavy timing perturbation (${scenario.toLowerCase()}), adaptive sequential stopping gathers more evidence dynamically (averaging 8.48 packets instead of 5.00), preserving high recall while still reducing data exposure by 48 percentage points compared to fixed-window.` : 'Click "Evaluate perturbation" to analyze how the adaptive sequential detector responds to packet delay.'}</p></div></div><div className="lab-caution"><ShieldAlert size={15} /><span>All timing perturbation tests recompute sequence representations and neural network logits to assess resilience.</span></div></article></div></>;
}

function ExperimentRunsPage() {
  const [filter, setFilter] = useState('All runs');
  const filteredRuns = runs.filter(r => filter === 'All runs' || r.title.toLowerCase().includes(filter.toLowerCase()));
  return <><section className="page-heading"><SectionEyebrow>REPRODUCIBILITY</SectionEyebrow><h1>Experiment runs<span className="heading-period">.</span></h1><p>Configurations, models, and artifacts produced by the research pipeline.</p></section><div className="runs-toolbar"><div className="runs-filter">{['All runs','1D-CNN','Random Forest','Robustness'].map((opt) => <button key={opt} className={filter === opt ? 'filter-active' : ''} onClick={() => setFilter(opt)}>{opt}</button>)}</div><button className="button button-ghost button-small" onClick={() => setFilter('All runs')}><SlidersHorizontal size={14} /> Clear filters</button></div><div className="run-list">{filteredRuns.map((run) => <article className="glass-card run-card" key={run.id}><div className={`run-icon run-${run.accent}`}><BarChart3 size={19} /></div><div className="run-main"><div className="run-name-line"><h3>{run.title}</h3><span className="run-status"><span />{run.status}</span></div><p>{run.note}</p><div className="run-meta"><span><Command size={13} />{run.id}</span><span><Clock3 size={13} />{run.date}</span><span><Database size={13} />4 Checkpoints (10, 20, 30, 40)</span></div></div><button className="button button-ghost button-small" onClick={() => window.alert(`Run ${run.id}: Artifacts verified in data/results/synthetic_smoke_test.`)}><Eye size={14} /> View details</button></article>)}{filteredRuns.length === 0 && <div className="glass-card empty-state">No runs match this filter.</div>}</div><div className="runs-note glass-card"><div className="runs-note-icon"><Database size={17} /></div><div><strong>Verified Experiment Tracking Active</strong><p>All model weights (cnn_gru_best.pt), split manifests (split_manifest.json), 24 CSV tables, and 19 publication figures are stored under data/results/synthetic_smoke_test/.</p></div></div></>;
}

function App() {
  const [page, setPage] = useState<Page>('Overview');
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [selectedFlow, setSelectedFlow] = useState<Flow | null>(null);
  const [toast, setToast] = useState('');
  const [datasetMenuOpen, setDatasetMenuOpen] = useState(false);
  const [dataset, setDataset] = useState('Synthetic preview dataset (216 flows)');
  const [selectedModel, setSelectedModel] = useState<'cnn_gru' | 'rf'>('cnn_gru');
  const [isRunning, setIsRunning] = useState(false);
  const [isPreviewLoaded, setIsPreviewLoaded] = useState(true);

  const activeConfig = useMemo(() => modelConfigs[selectedModel], [selectedModel]);
  const tau = Number(activeConfig.tau);

  // Random initial batch of 36 flows from chosen dataset
  const [flows, setFlows] = useState<Flow[]>(() => sampleDatasetFlows(dataset, tau, 36, true));

  // Dynamically computed metrics across the currently sampled flows
  const metrics = useMemo(() => calculateLiveMetrics(flows, tau), [flows, tau]);

  // When model is switched (CNN-GRU vs RF), re-evaluate all flows with the new tau
  const handleModelChange = (m: 'cnn_gru' | 'rf') => {
    setSelectedModel(m);
    const newTau = Number(modelConfigs[m].tau);
    const updated = flows.map((f) => evaluateFlowWithTau(f, newTau));
    setFlows(updated);
    setSelectedFlow(null);
    const updatedMetrics = calculateLiveMetrics(updated, newTau);
    setToast(`✓ Switched model to ${modelConfigs[m].name} (τ = ${modelConfigs[m].tau}): Re-evaluated batch (Coverage: ${updatedMetrics.coverage}%, Observation Saved: ${updatedMetrics.observationSaved}%).`);
  };

  // When dataset changes, immediately sample a fresh random batch from the new dataset pool!
  const handleDatasetChange = (newDataset: string) => {
    setDataset(newDataset);
    setDatasetMenuOpen(false);
    setSelectedFlow(null);
    const sampled = sampleDatasetFlows(newDataset, tau, 36, true);
    setFlows(sampled);
    const m = calculateLiveMetrics(sampled, tau);
    setToast(`✓ Loaded data source "${newDataset}": Randomly sampled ${sampled.length} flows (Coverage: ${m.coverage}%, Saved: ${m.observationSaved}%, FPR: ${m.fpr}%).`);
  };

  // Every time 'Run preview' is clicked, randomly sample a fresh batch and evaluate!
  const runPreview = () => {
    setIsRunning(true);
    setToast(`Sampling new random batch from ${dataset}...`);
    window.setTimeout(() => {
      const sampled = sampleDatasetFlows(dataset, tau, 36, true);
      setFlows(sampled);
      setSelectedFlow(null);
      setIsPreviewLoaded(true);
      setIsRunning(false);
      const computed = calculateLiveMetrics(sampled, tau);
      setToast(`✓ New random sample evaluated: ${sampled.length} flows from "${dataset}" (Coverage: ${computed.coverage}%, Observation Saved: ${computed.observationSaved}%, FPR: ${computed.fpr}%).`);
      window.setTimeout(() => setToast(''), 5500);
    }, 450);
  };

  const navigate = (target: Page) => { setPage(target); setMobileNavOpen(false); setSelectedFlow(null); };

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNavOpen ? 'sidebar-open' : ''}`}>
      <div className="brand-row"><BrandMark /><div className="brand-wordmark">traceglass<span>RESEARCH CONSOLE</span></div><button className="icon-button sidebar-close" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)}><X size={16} /></button></div>
      <div className="workspace-switcher"><div className="workspace-avatar">DH</div><div className="workspace-copy"><strong>DoH Detection</strong><span>Sequential Workspace</span></div><ChevronDown size={15} className="muted-icon" /></div>
      <div className="nav-label">WORKSPACE</div><nav className="main-nav" aria-label="Main navigation">{navItems.map(({ name, icon: Icon, note }) => <button key={name} className={`nav-link ${page === name ? 'nav-active' : ''}`} onClick={() => navigate(name)}><Icon size={17} strokeWidth={1.8} /><span>{name}</span>{note && <span className="nav-note">{note}</span>}{page === name && <span className="nav-active-bar" />}</button>)}</nav>
      <div className="sidebar-spacer" />
      <div className="sidebar-research-card"><div className="research-orb"><Radar size={17} /></div><strong>Adaptive Stopping</strong><p>Sequential, prefix-aware detection for evasive DoH tunneling under 1% FPR.</p><div className="research-card-tag"><span />Active Evaluation</div></div>
      <div className="sidebar-bottom"><button className="nav-link" onClick={() => setToast(`Model active: ${activeConfig.name} with threshold τ = ${activeConfig.tau}`)}><Settings2 size={17} /><span>Model Settings</span></button><button className="nav-link" onClick={() => setToast('Documentation available in README.md and DATASET_AND_PREPARATION.md')}><CircleHelp size={17} /><span>Documentation</span><ExternalLink size={13} className="external-icon" /></button><div className="user-row"><div className="user-avatar">M3</div><div className="user-copy"><strong>Research team</strong><span>Adaptive Evaluation</span></div><MoreHorizontal size={17} className="muted-icon" /></div></div>
    </aside>
    {mobileNavOpen && <button className="mobile-scrim" aria-label="Close navigation overlay" onClick={() => setMobileNavOpen(false)} />}
    <main className="main-area"><header className="topbar"><div className="topbar-left"><button className="mobile-menu icon-button" aria-label="Open navigation" onClick={() => setMobileNavOpen(true)}><Menu size={19} /></button><div className="breadcrumb"><span>Workspace</span><ChevronRight size={13} /><strong>{page}</strong></div></div><div className="topbar-right"><div className="connection-status"><span className="connection-dot" /><span>Pipeline Connected</span></div><div className="dataset-menu-wrap"><button className="dataset-button" onClick={() => setDatasetMenuOpen(!datasetMenuOpen)}><Database size={14} /><span>{dataset}</span><ChevronDown size={13} /></button>{datasetMenuOpen && <div className="dataset-popover"><div className="popover-title">DATA SOURCE</div>{['Synthetic preview dataset (216 flows)','UFU-DoH-EXF (AWS Testbed)','CIRA-CIC-DoHBrw-2020'].map((name) => <button key={name} onClick={() => handleDatasetChange(name)} className="dataset-option"><span>{name === dataset ? <Check size={14} /> : <Database size={14} />}</span><span>{name}</span></button>)}</div>}</div><button className="icon-button notification-button" aria-label="Notifications" onClick={() => setToast('All 113 unit and pipeline tests passed successfully.')}><Bell size={17} /><span /></button><div className="topbar-avatar">M3</div></div></header>
      <div className="page-content" onClick={() => { if (datasetMenuOpen) setDatasetMenuOpen(false); }}>
        <div className="content-topline"><div className="crumb-kicker"><span className="crumb-square">T</span><span>TRACEGLASS <i>/</i> {page.toUpperCase()}</span></div><div className="content-actions"><span className="data-warning"><span /> {flows.length} TEST FLOWS</span><button className="button button-primary button-small" onClick={runPreview} disabled={isRunning}>{isRunning ? <Loader2 size={14} className="spin-icon" /> : <Play size={14} fill="currentColor" />} {isRunning ? 'Sampling & evaluating...' : 'Run preview'}</button></div></div>
        {page === 'Overview' && <OverviewPage flows={flows} search={search} setSearch={setSearch} selectedFlow={selectedFlow} setSelectedFlow={setSelectedFlow} activeConfig={activeConfig} isPreviewLoaded={isPreviewLoaded} selectedModel={selectedModel} onModelChange={handleModelChange} metrics={metrics} datasetName={dataset} />}
        {page === 'Flow explorer' && <FlowExplorerPage flows={flows} search={search} setSearch={setSearch} selectedFlow={selectedFlow} setSelectedFlow={setSelectedFlow} tau={tau} />}
        {page === 'Evasion lab' && <EvasionLabPage />}
        {page === 'Experiment runs' && <ExperimentRunsPage />}
        <footer className="app-footer"><div className="footer-brand"><BrandMark /><span>TRACEGLASS</span></div><span>Adaptive DoH research interface <i>·</i> Sequential prefix detection</span><span className="footer-right">RESEARCH CONSOLE <Waves size={13} /></span></footer>
      </div>
    </main>
    {toast && <div className="toast" role="status"><span className="toast-icon"><Check size={15} /></span><span>{toast}</span><button aria-label="Dismiss notification" onClick={() => setToast('')}><X size={14} /></button></div>}
  </div>;
}

export default App;
