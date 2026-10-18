import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import {
  Activity, ArrowRight, ArrowUpRight, Atom, Bell, Check,
  BarChart3, ChevronDown, ChevronRight, CircleHelp, Clock3, Command, Database,
  ExternalLink, Eye, FlaskConical, Gauge, GitCompare, Layers3, LockKeyhole, Menu, MoreHorizontal, Network,
  Play, Radar, Search, Settings2, ShieldAlert, ShieldCheck, SlidersHorizontal,
  Sparkles, Timer, Waves, X, Zap,
} from 'lucide-react';

type Page = 'Overview' | 'Flow explorer' | 'Evasion lab' | 'Experiment runs';
type Flow = {
  id: string;
  label: 'Malicious' | 'Benign';
  score: number;
  checkpoint: string;
  bytes: string;
  status: 'Early alert' | 'Final checkpoint' | 'No alert';
  duration: string;
  source: string;
  note: string;
};

type NavItem = { name: Page; icon: LucideIcon; note?: string };

const navItems: NavItem[] = [
  { name: 'Overview', icon: Layers3 },
  { name: 'Flow explorer', icon: Network },
  { name: 'Evasion lab', icon: FlaskConical, note: 'LAB' },
  { name: 'Experiment runs', icon: BarChart3 },
];

const initialFlows: Flow[] = [
  { id: 'FLW-8F2A', label: 'Malicious', score: 0.94, checkpoint: '30%', bytes: '1.8 KB', status: 'Early alert', duration: '0.84 s', source: '192.0.2.48', note: 'The score crossed the illustrative threshold at checkpoint 3 of 4.' },
  { id: 'FLW-01C9', label: 'Benign', score: 0.12, checkpoint: '40%', bytes: '4.1 KB', status: 'No alert', duration: '2.11 s', source: '192.0.2.117', note: 'No checkpoint crossed the illustrative alert threshold.' },
  { id: 'FLW-72BD', label: 'Malicious', score: 0.88, checkpoint: '20%', bytes: '0.9 KB', status: 'Early alert', duration: '0.39 s', source: '192.0.2.206', note: 'An early checkpoint crossed the threshold in this mock scenario.' },
  { id: 'FLW-44E0', label: 'Benign', score: 0.67, checkpoint: '40%', bytes: '3.6 KB', status: 'No alert', duration: '1.76 s', source: '192.0.2.89', note: 'The score remained below the selected mock threshold.' },
  { id: 'FLW-9A11', label: 'Malicious', score: 0.81, checkpoint: '40%', bytes: '2.7 KB', status: 'Final checkpoint', duration: '1.42 s', source: '192.0.2.14', note: 'The threshold was crossed only at the final observation checkpoint.' },
];

const runs = [
  { id: 'RUN-DEMO-014', title: 'Adaptive policy · 4 checkpoints', date: 'Today · Demo session', status: 'Preview', accent: 'mint', note: 'Sequential stopping policy with synthetic flow curves.' },
  { id: 'RUN-DEMO-013', title: 'Fixed-window baseline', date: 'Today · Demo session', status: 'Preview', accent: 'teal', note: 'All flows observed up to the selected fixed checkpoint.' },
  { id: 'RUN-DEMO-012', title: 'Low-and-slow perturbation', date: 'Yesterday · Demo session', status: 'Preview', accent: 'foam', note: 'Illustrative timing perturbation; not a live traffic test.' },
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
      <div className="chart-legend"><span><i className="legend-dot legend-mint" />Adaptive stopping</span><span><i className="legend-dot legend-blue" />Fixed-window</span><span className="legend-context">Illustrative curves</span></div>
      <svg className="detection-chart" viewBox="0 0 650 244" role="img" aria-label="Illustrative detection progression comparing adaptive and fixed-window policies">
        <defs>
          <linearGradient id="mintFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#9fffc8" stopOpacity=".18"/><stop offset="100%" stopColor="#9fffc8" stopOpacity="0"/></linearGradient>
          <linearGradient id="blueFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#66c9f2" stopOpacity=".12"/><stop offset="100%" stopColor="#66c9f2" stopOpacity="0"/></linearGradient>
        </defs>
        {[32, 76, 120, 164, 208].map((y) => <line key={y} x1="42" y1={y} x2="625" y2={y} className="grid-line" />)}
        {[42, 188, 334, 480, 625].map((x) => <line key={x} x1={x} y1="25" x2={x} y2="208" className="grid-line vertical-grid" />)}
        <text x="8" y="36" className="axis-label">100</text><text x="14" y="80" className="axis-label">75</text><text x="14" y="124" className="axis-label">50</text><text x="14" y="168" className="axis-label">25</text><text x="20" y="212" className="axis-label">0</text>
        <path d="M42 180 C91 155 133 130 188 112 S278 74 334 67 S428 53 480 48 S576 41 625 36 L625 208 L42 208 Z" fill="url(#mintFill)" />
        <path d="M42 180 C91 155 133 130 188 112 S278 74 334 67 S428 53 480 48 S576 41 625 36" className="chart-path path-mint" />
        <path d="M42 189 C104 180 146 160 188 145 S285 121 334 101 S430 85 480 67 S570 56 625 49 L625 208 L42 208 Z" fill="url(#blueFill)" />
        <path d="M42 189 C104 180 146 160 188 145 S285 121 334 101 S430 85 480 67 S570 56 625 49" className="chart-path path-blue" />
        {[[42,180],[188,112],[334,67],[480,48],[625,36]].map(([x,y]) => <circle key={x} cx={x} cy={y} r="4.2" className="chart-point point-mint" />)}
        {[[42,189],[188,145],[334,101],[480,67],[625,49]].map(([x,y]) => <circle key={x} cx={x} cy={y} r="3.6" className="chart-point point-blue" />)}
        <text x="42" y="231" className="axis-bottom" textAnchor="middle">Start</text><text x="188" y="231" className="axis-bottom" textAnchor="middle">10%</text><text x="334" y="231" className="axis-bottom" textAnchor="middle">20%</text><text x="480" y="231" className="axis-bottom" textAnchor="middle">30%</text><text x="625" y="231" className="axis-bottom" textAnchor="middle">40%</text>
      </svg>
      <div className="chart-note"><span className="chart-note-icon"><Activity size={14} /></span><span>Illustrative trend only. Real curves will appear after experiment results are connected.</span></div>
    </div>
  );
}

function CheckpointRail() {
  const checkpoints = [
    { step: '01', pct: '10%', name: 'First look', width: '24%', type: 'observe' },
    { step: '02', pct: '20%', name: 'Reassess', width: '42%', type: 'observe' },
    { step: '03', pct: '30%', name: 'Decision zone', width: '70%', type: 'alert' },
    { step: '04', pct: '40%', name: 'Final window', width: '100%', type: 'final' },
  ];
  return <div className="checkpoint-list">{checkpoints.map((c) => <div className="checkpoint-row" key={c.step}><div className="checkpoint-index">{c.step}</div><div className="checkpoint-main"><div className="checkpoint-copy"><strong>{c.pct}</strong><span>{c.name}</span></div><div className="checkpoint-track"><span className={`checkpoint-fill fill-${c.type}`} style={{ width: c.width }} /></div></div><span className={`checkpoint-state state-${c.type}`}>{c.type === 'alert' ? 'Threshold' : c.type === 'final' ? 'Max prefix' : 'Observe'}</span></div>)}</div>;
}

function FlowTable({ flows, search, onSearch, selectedId, onSelect }: {
  flows: Flow[]; search: string; onSearch: (s: string) => void; selectedId: string | null; onSelect: (id: string) => void;
}) {
  const filtered = flows.filter((flow) => `${flow.id} ${flow.label} ${flow.status} ${flow.source}`.toLowerCase().includes(search.toLowerCase()));
  return (
    <div className="table-card glass-card">
      <div className="table-head"><div><div className="panel-title">Recent flow decisions</div><div className="panel-subtitle">A small synthetic sample to preview the investigation view.</div></div><div className="table-actions"><div className="search-field"><Search size={14} /><input aria-label="Search flows" placeholder="Search flows..." value={search} onChange={(e) => onSearch(e.target.value)} /></div><button className="button button-ghost button-small" onClick={() => onSearch('')}><SlidersHorizontal size={14} /> Reset</button></div></div>
      <div className="flow-table-scroll"><table className="flow-table"><thead><tr><th>Flow ID</th><th>Ground truth</th><th>Score</th><th>Stopped at</th><th>Observation</th><th>Decision</th><th /></tr></thead><tbody>
        {filtered.map((flow) => <tr key={flow.id} className={selectedId === flow.id ? 'selected-row' : ''} onClick={() => onSelect(flow.id)} tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') onSelect(flow.id); }}>
          <td><span className="flow-id"><span className="flow-id-dot" />{flow.id}</span></td><td><span className={`label-pill label-${flow.label.toLowerCase()}`}>{flow.label}</span></td><td><div className="score-cell"><span>{flow.score.toFixed(2)}</span><div className="score-track"><span style={{ width: `${flow.score * 100}%` }} className={flow.label === 'Malicious' ? 'score-malicious' : 'score-benign'} /></div></div></td><td className="mono-cell">{flow.checkpoint}</td><td className="mono-cell">{flow.bytes}</td><td><span className={`decision decision-${flow.status.toLowerCase().replace(' ', '-')}`}><span />{flow.status}</span></td><td><ChevronRight size={15} className="row-chevron" /></td>
        </tr>)}
      </tbody></table>{filtered.length === 0 && <div className="empty-state">No demo flows match “{search}”. Try another search.</div>}</div>
      <div className="table-footer"><span>Showing {filtered.length} of {flows.length} synthetic flows</span><button className="text-button" onClick={() => onSelect(filtered[0]?.id ?? '')}>Open flow explorer <ArrowRight size={13} /></button></div>
    </div>
  );
}

function FlowDetail({ flow, onClose }: { flow: Flow; onClose: () => void }) {
  return <div className="detail-panel glass-card"><div className="detail-head"><div><SectionEyebrow>FLOW INSPECTOR</SectionEyebrow><h3>{flow.id}</h3></div><button className="icon-button" aria-label="Close flow detail" onClick={onClose}><X size={16} /></button></div><div className="detail-badges"><span className={`label-pill label-${flow.label.toLowerCase()}`}>{flow.label} · Ground truth</span><span className="tag tag-outline">Synthetic</span></div><div className="detail-score"><div><span>Final demo score</span><strong>{flow.score.toFixed(2)}</strong></div><div className="score-track score-track-lg"><span style={{ width: `${flow.score * 100}%` }} className={flow.label === 'Malicious' ? 'score-malicious' : 'score-benign'} /></div></div><div className="detail-stats"><div><span>Observed prefix</span><strong>{flow.checkpoint}</strong></div><div><span>Bytes observed</span><strong>{flow.bytes}</strong></div><div><span>Elapsed time</span><strong>{flow.duration}</strong></div><div><span>Example source</span><strong>{flow.source}</strong></div></div><div className="detail-explanation"><div className="explanation-title"><Sparkles size={14} /> Decision trace</div><p>{flow.note}</p><div className="trace-steps"><span className="trace-complete"><Check size={12} /> Prefix built</span><span className="trace-complete"><Check size={12} /> Score computed</span><span className={flow.status === 'No alert' ? 'trace-pending' : 'trace-complete'}>{flow.status === 'No alert' ? <Clock3 size={12} /> : <ShieldAlert size={12} />} {flow.status === 'No alert' ? 'No threshold crossing' : 'Decision reached'}</span></div></div></div>;
}

function OverviewPage({ flows, search, setSearch, selectedFlow, setSelectedFlow }: {
  flows: Flow[]; search: string; setSearch: (s: string) => void; selectedFlow: Flow | null; setSelectedFlow: (flow: Flow | null) => void;
}) {
  return <>
    <section className="hero-row"><div><SectionEyebrow>SEQUENTIAL DETECTION WORKSPACE</SectionEyebrow><h1>See the signal.<br /><span>Stop at the right moment.</span></h1><p className="hero-copy">Explore how adaptive stopping can surface DoH tunneling earlier — without observing more traffic than the decision needs.</p><div className="hero-footnote"><span className="mini-wave"><Waves size={16} /></span><span>Research preview <b>·</b> Synthetic data <b>·</b> No live traffic connected</span></div></div><div className="hero-visual" aria-hidden="true"><div className="visual-orbit orbit-one" /><div className="visual-orbit orbit-two" /><div className="visual-core"><Radar size={36} strokeWidth={1.3} /></div><div className="visual-node node-one"><span /><i>FLOW</i></div><div className="visual-node node-two"><span /><i>DETECT</i></div><div className="visual-node node-three"><span /><i>STOP</i></div><div className="visual-ring-caption">PREFIX STREAM <span>01 — 04</span></div></div></section>
    <div className="preview-banner"><div className="preview-banner-icon"><Atom size={17} /></div><div><strong>You're viewing the UI preview</strong><span>Every metric, flow, and chart on this screen is illustrative mock data — not experimental evidence.</span></div><span className="preview-badge">SYNTHETIC PREVIEW</span></div>
    <section className="metric-grid"><MetricCard icon={ShieldCheck} title="Detection coverage" value="91.4" unit="%" change="+8.2 pts" changeUp tone="mint" detail="vs. fixed-window*" /><MetricCard icon={Timer} title="Median stop point" value="27" unit="%" change="−13 pts" changeUp tone="teal" detail="traffic observed*" /><MetricCard icon={Gauge} title="Observation saved" value="38.6" unit="%" change="vs. full flow" tone="foam" detail="bytes / prefix proxy*" /><MetricCard icon={ShieldAlert} title="False-positive rate" value="0.8" unit="%" change="Target ≤ 1.0%" tone="blue" detail="validation policy*" /></section>
    <section className="analysis-grid"><article className="glass-card chart-card"><div className="panel-header"><div><SectionEyebrow>POLICY COMPARISON</SectionEyebrow><h2>Detection progression</h2><p>Illustrative detection score trajectory as each prefix arrives.</p></div><button className="icon-button" aria-label="Chart options"><MoreHorizontal size={18} /></button></div><DetectionChart /></article><article className="glass-card stopping-card"><div className="panel-header"><div><SectionEyebrow color="teal">THE STOPPING RULE</SectionEyebrow><h2>Observe only as needed</h2><p>Score each prefix, then stop when the alert threshold is crossed.</p></div><div className="stop-icon"><Zap size={17} /></div></div><div className="threshold-box"><div className="threshold-heading"><span>Alert threshold <span className="threshold-example">DEMO</span></span><strong>τ = 0.82</strong></div><div className="threshold-rail"><span /></div><div className="threshold-scale"><span>0.00 · lower confidence</span><span>1.00 · higher</span></div><div className="threshold-status"><span className="status-indicator" /><span>Validation-derived threshold</span><span className="status-demo">Example only</span></div></div><CheckpointRail /><div className="policy-footer"><div className="policy-footer-icon"><LockKeyhole size={14} /></div><span>Threshold selection should use validation flows — never the held-out test set.</span></div></article></section>
    <section className="flow-section"><FlowTable flows={flows} search={search} onSearch={setSearch} selectedId={selectedFlow?.id ?? null} onSelect={(id) => setSelectedFlow(flows.find((f) => f.id === id) ?? null)} />{selectedFlow && <FlowDetail flow={selectedFlow} onClose={() => setSelectedFlow(null)} />}</section>
    <div className="footnote">* Preview-only figures. Replace with results from the verified evaluation pipeline before presenting or reporting research findings.</div>
  </>;
}

function FlowExplorerPage({ flows, search, setSearch, selectedFlow, setSelectedFlow }: {
  flows: Flow[]; search: string; setSearch: (s: string) => void; selectedFlow: Flow | null; setSelectedFlow: (flow: Flow | null) => void;
}) {
  return <><section className="page-heading"><SectionEyebrow>INVESTIGATE THE DECISION</SectionEyebrow><h1>Flow explorer<span className="heading-period">.</span></h1><p>Trace the stopping point, score and observation footprint for an individual flow.</p></section><div className="explorer-stats"><div className="glass-card compact-stat"><span>Flows in preview</span><strong>{flows.length.toString().padStart(2, '0')}</strong></div><div className="glass-card compact-stat"><span>Early alerts</span><strong>{flows.filter(f => f.status === 'Early alert').length.toString().padStart(2, '0')}</strong></div><div className="glass-card compact-stat"><span>Unresolved / no alert</span><strong>{flows.filter(f => f.status === 'No alert').length.toString().padStart(2, '0')}</strong></div></div><FlowTable flows={flows} search={search} onSearch={setSearch} selectedId={selectedFlow?.id ?? null} onSelect={(id) => setSelectedFlow(flows.find((f) => f.id === id) ?? null)} />{selectedFlow ? <FlowDetail flow={selectedFlow} onClose={() => setSelectedFlow(null)} /> : <div className="glass-card empty-selection"><Eye size={18} /><strong>Select a flow to inspect its decision trace</strong><span>Use the table above to view checkpoint and observation details.</span></div>}<div className="footnote">All rows and trace details on this page are synthetic examples. No PCAP data is being read.</div></>;
}

function EvasionLabPage() {
  const [scenario, setScenario] = useState('Low-and-slow');
  const [intensity, setIntensity] = useState(45);
  const [jitter, setJitter] = useState(20);
  const [previewed, setPreviewed] = useState(false);
  return <><section className="page-heading"><SectionEyebrow color="teal">OFFLINE SCENARIO LAB</SectionEyebrow><h1>Evasion lab<span className="heading-period">.</span></h1><p>Explore how timing perturbations may affect sequential detection. This is a mock control surface, not a traffic generator.</p></section><div className="lab-grid"><article className="glass-card lab-controls"><div className="panel-header"><div><h2>Scenario configuration</h2><p>Adjust the illustrative perturbation settings.</p></div><div className="lab-badge"><FlaskConical size={14} /> SIMULATED</div></div><label className="field-label">Scenario type</label><div className="scenario-options">{[{name:'Low-and-slow',desc:'Longer gaps between packets',icon:Clock3},{name:'Timing jitter',desc:'Vary packet arrival timing',icon:Activity},{name:'Combined',desc:'Slow rate + timing variation',icon:GitCompare}].map((option) => { const Icon = option.icon; return <button key={option.name} className={`scenario-option ${scenario === option.name ? 'scenario-selected' : ''}`} onClick={() => {setScenario(option.name); setPreviewed(false);}}><span className="scenario-option-icon"><Icon size={17} /></span><span className="scenario-option-copy"><strong>{option.name}</strong><small>{option.desc}</small></span><span className="radio-mark">{scenario === option.name && <span />}</span></button>;})}</div><div className="range-block"><div className="range-label"><label htmlFor="intensity">Pacing intensity</label><strong>{intensity}<small> / 100</small></strong></div><input id="intensity" type="range" min="0" max="100" value={intensity} onChange={(e) => {setIntensity(Number(e.target.value)); setPreviewed(false);}} /><div className="range-ends"><span>Normal</span><span>Very slow</span></div></div><div className="range-block"><div className="range-label"><label htmlFor="jitter">Timing jitter</label><strong>{jitter}<small> %</small></strong></div><input id="jitter" type="range" min="0" max="100" value={jitter} onChange={(e) => {setJitter(Number(e.target.value)); setPreviewed(false);}} /><div className="range-ends"><span>None</span><span>High variation</span></div></div><button className="button button-primary full-button" onClick={() => setPreviewed(true)}><Play size={15} fill="currentColor" /> Preview scenario <ArrowRight size={15} /></button><p className="control-disclaimer"><LockKeyhole size={13} /> Does not access the network or run a real model.</p></article><article className="glass-card lab-results"><div className="panel-header"><div><SectionEyebrow color="foam">SCENARIO READOUT</SectionEyebrow><h2>{scenario}</h2><p>{previewed ? 'Illustrative scenario parameters updated.' : 'Preview readout for your selected configuration.'}</p></div><div className="scenario-wave"><Waves size={24} /></div></div><div className="readout-visual"><div className="readout-axis"><span>EARLIER</span><span>LATER</span></div><div className="readout-wave"><div className="wave-base" /><svg viewBox="0 0 560 120" preserveAspectRatio="none" aria-label="Illustrative timing pattern"><path d={`M0 60 ${Array.from({length: 42}, (_, i) => { const x = (i + 1) * 13.3; const y = 60 + Math.sin(i * (scenario === 'Timing jitter' ? 2.8 : 1.15)) * (scenario === 'Low-and-slow' ? intensity * .34 : jitter * .45); return `L${x.toFixed(1)} ${Math.max(10, Math.min(110, y)).toFixed(1)}`; }).join(' ')}`} className="scenario-wave-path" /></svg></div><div className="readout-markers"><span>Prefix 01</span><span>Prefix 02</span><span>Prefix 03</span><span>Prefix 04</span></div></div><div className="scenario-readout-cards"><div><span>Scenario</span><strong>{scenario}</strong></div><div><span>Pacing</span><strong>{intensity}%</strong></div><div><span>Jitter</span><strong>{jitter}%</strong></div></div><div className={`readout-callout ${previewed ? 'readout-updated' : ''}`}><div className="readout-callout-icon"><Activity size={16} /></div><div><strong>{previewed ? 'Preview updated' : 'Awaiting preview'}</strong><p>{previewed ? `This mock preview represents ${scenario.toLowerCase()} settings with pacing ${intensity}/100 and timing jitter ${jitter}%. Run the real evaluation pipeline to measure impact.` : 'Change the settings and preview a scenario. Detection metrics are intentionally not fabricated from these sliders.'}</p></div></div><div className="lab-caution"><ShieldAlert size={15} /><span>Keep perturbation results separate from real-world evasion claims. Report them as offline simulated timing scenarios.</span></div></article></div></>;
}

function ExperimentRunsPage() {
  const [filter, setFilter] = useState('All runs');
  const filteredRuns = runs.filter(r => filter === 'All runs' || r.title.toLowerCase().includes(filter.toLowerCase()));
  return <><section className="page-heading"><SectionEyebrow>REPRODUCIBILITY</SectionEyebrow><h1>Experiment runs<span className="heading-period">.</span></h1><p>Track the configurations and outputs produced by the research pipeline.</p></section><div className="runs-toolbar"><div className="runs-filter">{['All runs','Adaptive','Fixed-window','Low-and-slow'].map((opt) => <button key={opt} className={filter === opt ? 'filter-active' : ''} onClick={() => setFilter(opt)}>{opt}</button>)}</div><button className="button button-ghost button-small" onClick={() => setFilter('All runs')}><SlidersHorizontal size={14} /> Clear filters</button></div><div className="run-list">{filteredRuns.map((run) => <article className="glass-card run-card" key={run.id}><div className={`run-icon run-${run.accent}`}><BarChart3 size={19} /></div><div className="run-main"><div className="run-name-line"><h3>{run.title}</h3><span className="run-status"><span />{run.status}</span></div><p>{run.note}</p><div className="run-meta"><span><Command size={13} />{run.id}</span><span><Clock3 size={13} />{run.date}</span><span><Database size={13} />Synthetic sample</span></div></div><button className="button button-ghost button-small" onClick={() => window.alert('This is a UI preview. Connect the experiment API to load run artifacts.')}><Eye size={14} /> View details</button></article>)}{filteredRuns.length === 0 && <div className="glass-card empty-state">No runs match this filter in the synthetic preview.</div>}</div><div className="runs-note glass-card"><div className="runs-note-icon"><Database size={17} /></div><div><strong>Real experiment tracking is not connected yet.</strong><p>Once the evaluation pipeline emits a run manifest, this page can display its config, data split summary, metrics, figures and artifact paths.</p></div></div></>;
}

function App() {
  const [page, setPage] = useState<Page>('Overview');
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [selectedFlow, setSelectedFlow] = useState<Flow | null>(null);
  const [toast, setToast] = useState('');
  const [datasetMenuOpen, setDatasetMenuOpen] = useState(false);
  const [dataset, setDataset] = useState('Synthetic preview dataset');
  const filteredPageFlows = useMemo(() => initialFlows, []);

  const runPreview = () => {
    setToast('Preview ready · No model or PCAP files were executed.');
    window.setTimeout(() => setToast(''), 3800);
  };
  const navigate = (target: Page) => { setPage(target); setMobileNavOpen(false); setSelectedFlow(null); };

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNavOpen ? 'sidebar-open' : ''}`}>
      <div className="brand-row"><BrandMark /><div className="brand-wordmark">traceglass<span>RESEARCH CONSOLE</span></div><button className="icon-button sidebar-close" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)}><X size={16} /></button></div>
      <div className="workspace-switcher"><div className="workspace-avatar">DH</div><div className="workspace-copy"><strong>DoH Detection</strong><span>Research workspace</span></div><ChevronDown size={15} className="muted-icon" /></div>
      <div className="nav-label">WORKSPACE</div><nav className="main-nav" aria-label="Main navigation">{navItems.map(({ name, icon: Icon, note }) => <button key={name} className={`nav-link ${page === name ? 'nav-active' : ''}`} onClick={() => navigate(name)}><Icon size={17} strokeWidth={1.8} /><span>{name}</span>{note && <span className="nav-note">{note}</span>}{page === name && <span className="nav-active-bar" />}</button>)}</nav>
      <div className="sidebar-spacer" />
      <div className="sidebar-research-card"><div className="research-orb"><Radar size={17} /></div><strong>Research mode</strong><p>Sequential, flow-aware detection for evasive DoH tunneling.</p><div className="research-card-tag"><span />Offline preview</div></div>
      <div className="sidebar-bottom"><button className="nav-link" onClick={() => setToast('Settings are planned for a later integration step.')}><Settings2 size={17} /><span>Workspace settings</span></button><button className="nav-link" onClick={() => setToast('Connect the backend to load real experiment data.')}><CircleHelp size={17} /><span>Documentation</span><ExternalLink size={13} className="external-icon" /></button><div className="user-row"><div className="user-avatar">M3</div><div className="user-copy"><strong>Research team</strong><span>Member 03 · Evaluation</span></div><MoreHorizontal size={17} className="muted-icon" /></div></div>
    </aside>
    {mobileNavOpen && <button className="mobile-scrim" aria-label="Close navigation overlay" onClick={() => setMobileNavOpen(false)} />}
    <main className="main-area"><header className="topbar"><div className="topbar-left"><button className="mobile-menu icon-button" aria-label="Open navigation" onClick={() => setMobileNavOpen(true)}><Menu size={19} /></button><div className="breadcrumb"><span>Workspace</span><ChevronRight size={13} /><strong>{page}</strong></div></div><div className="topbar-right"><div className="connection-status"><span className="connection-dot" /><span>Demo environment</span></div><div className="dataset-menu-wrap"><button className="dataset-button" onClick={() => setDatasetMenuOpen(!datasetMenuOpen)}><Database size={14} /><span>{dataset}</span><ChevronDown size={13} /></button>{datasetMenuOpen && <div className="dataset-popover"><div className="popover-title">DATA SOURCE</div>{['Synthetic preview dataset','CIRA-CIC-DoHBrw-2020','UFU-DoH-EXF'].map((name) => <button key={name} onClick={() => {setDataset(name);setDatasetMenuOpen(false);if(name !== 'Synthetic preview dataset') setToast('Dataset selection is a UI placeholder; no files have been loaded.');}} className="dataset-option"><span>{name === dataset ? <Check size={14} /> : <Database size={14} />}</span><span>{name}</span>{name !== 'Synthetic preview dataset' && <small>Not connected</small>}</button>)}</div>}</div><button className="icon-button notification-button" aria-label="Notifications" onClick={() => setToast('No new notifications.')}><Bell size={17} /><span /></button><div className="topbar-avatar">M3</div></div></header>
      <div className="page-content" onClick={() => { if (datasetMenuOpen) setDatasetMenuOpen(false); }}>
        <div className="content-topline"><div className="crumb-kicker"><span className="crumb-square">T</span><span>TRACEGLASS <i>/</i> {page.toUpperCase()}</span></div><div className="content-actions"><span className="data-warning"><span /> SAMPLE DATA</span><button className="button button-primary button-small" onClick={runPreview}><Play size={14} fill="currentColor" /> Run preview</button></div></div>
        {page === 'Overview' && <OverviewPage flows={filteredPageFlows} search={search} setSearch={setSearch} selectedFlow={selectedFlow} setSelectedFlow={setSelectedFlow} />}
        {page === 'Flow explorer' && <FlowExplorerPage flows={filteredPageFlows} search={search} setSearch={setSearch} selectedFlow={selectedFlow} setSelectedFlow={setSelectedFlow} />}
        {page === 'Evasion lab' && <EvasionLabPage />}
        {page === 'Experiment runs' && <ExperimentRunsPage />}
        <footer className="app-footer"><div className="footer-brand"><BrandMark /><span>TRACEGLASS</span></div><span>Adaptive DoH research interface <i>·</i> UI preview build</span><span className="footer-right">BUILT FOR INVESTIGATION <Waves size={13} /></span></footer>
      </div>
    </main>
    {toast && <div className="toast" role="status"><span className="toast-icon"><Check size={15} /></span><span>{toast}</span><button aria-label="Dismiss notification" onClick={() => setToast('')}><X size={14} /></button></div>}
  </div>;
}

export default App;
