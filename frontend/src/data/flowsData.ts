import allDatasetsRaw from './allDatasets.json';

export type Flow = {
  id: string;
  label: 'Malicious' | 'Benign';
  score: number;
  checkpoint: string;
  bytes: string;
  status: 'Early alert' | 'Final checkpoint' | 'No alert';
  duration: string;
  source: string;
  note: string;
  p10: number;
  p20: number;
  p30: number;
  p40: number;
  stopPkts: number;
};

export const allDatasets: Record<string, any[]> = allDatasetsRaw as Record<string, any[]>;

export type ModelConfig = {
  name: string;
  tau: string;
  baselineCoverage: string;
  baselineStopPoint: string;
  baselineObservationSaved: string;
  baselineFpr: string;
};

export const modelConfigs: Record<'cnn_gru' | 'rf', ModelConfig> = {
  cnn_gru: {
    name: '1D-CNN + GRU (Sequential Adaptive)',
    tau: '0.924',
    baselineCoverage: '98.2',
    baselineStopPoint: '25',
    baselineObservationSaved: '31.3',
    baselineFpr: '0.91',
  },
  rf: {
    name: 'Random Forest Baseline',
    tau: '0.814',
    baselineCoverage: '96.3',
    baselineStopPoint: '35',
    baselineObservationSaved: '30.6',
    baselineFpr: '0.95',
  },
};

export function evaluateFlowWithTau(flow: any, tau: number): Flow {
  const p10 = Number(flow.p10 ?? 0);
  const p20 = Number(flow.p20 ?? 0);
  const p30 = Number(flow.p30 ?? 0);
  const p40 = Number(flow.p40 ?? 0);

  let status: 'Early alert' | 'Final checkpoint' | 'No alert';
  let checkpoint: string;
  let score: number;
  let stopPkts: number;
  let note = flow.note || '';

  if (p10 >= tau) {
    status = 'Early alert';
    checkpoint = '10 pkts (25%)';
    score = p10;
    stopPkts = 10;
    note = `Immediate alert declared at prefix t=10 pkts (p=${p10.toFixed(3)} ≥ τ ${tau.toFixed(3)}). Pre-alert observation reduced by 75%.`;
  } else if (p20 >= tau) {
    status = 'Early alert';
    checkpoint = '20 pkts (50%)';
    score = p20;
    stopPkts = 20;
    note = `Rapid detection at checkpoint t=20 pkts (p=${p20.toFixed(3)} ≥ τ ${tau.toFixed(3)}). Pre-alert observation reduced by 50%.`;
  } else if (p30 >= tau) {
    status = 'Early alert';
    checkpoint = '30 pkts (75%)';
    score = p30;
    stopPkts = 30;
    note = `Confirmed tunneling alert at checkpoint t=30 pkts (p=${p30.toFixed(3)} ≥ τ ${tau.toFixed(3)}). Pre-alert observation reduced by 25%.`;
  } else if (p40 >= tau) {
    status = 'Final checkpoint';
    checkpoint = '40 pkts (100%)';
    score = p40;
    stopPkts = 40;
    note = `Threshold crossed at final observation window t=40 pkts (p=${p40.toFixed(3)} ≥ τ ${tau.toFixed(3)}). Full window inspected.`;
  } else {
    status = 'No alert';
    checkpoint = '40 pkts (100%)';
    score = p40;
    stopPkts = 40;
    note = `Calibrated score remained below alert threshold τ=${tau.toFixed(3)} across all 4 checkpoints. Safe benign traffic declared.`;
  }

  return {
    id: flow.id,
    label: flow.label,
    bytes: flow.bytes || '2.4 KB',
    duration: flow.duration || '0.95 s',
    source: flow.source || '192.168.10.1',
    score,
    checkpoint,
    status,
    note,
    p10,
    p20,
    p30,
    p40,
    stopPkts,
  };
}

export function sampleDatasetFlows(
  datasetName: string,
  tau: number,
  sampleSize = 36,
  randomize = true
): Flow[] {
  const pool = allDatasets[datasetName] ?? allDatasets['Synthetic preview dataset (216 flows)'];
  if (!pool || pool.length === 0) return [];

  let selected: any[];
  if (randomize) {
    const shuffled = [...pool].sort(() => Math.random() - 0.5);
    selected = shuffled.slice(0, Math.min(sampleSize, pool.length));
  } else {
    selected = pool.slice(0, Math.min(sampleSize, pool.length));
  }

  return selected.map((f) => evaluateFlowWithTau(f, tau));
}

export type LiveMetrics = {
  coverage: string;
  coverageDetail: string;
  stopPoint: string;
  stopPointDetail: string;
  observationSaved: string;
  observationDetail: string;
  fpr: string;
  fprDetail: string;
  earlyAlertCount: number;
  unresolvedCount: number;
  totalEvaluated: number;
  maliciousCount: number;
  detectedMaliciousCount: number;
  benignCount: number;
  fpCount: number;
};

export function calculateLiveMetrics(flows: Flow[], _tau?: number): LiveMetrics {
  const malicious = flows.filter((f) => f.label === 'Malicious');
  const benign = flows.filter((f) => f.label === 'Benign');

  const detected = malicious.filter((f) => f.status !== 'No alert');
  const coverageNum = malicious.length > 0 ? (detected.length / malicious.length) * 100 : 100;

  const falsePositives = benign.filter((f) => f.status !== 'No alert');
  const fprNum = benign.length > 0 ? (falsePositives.length / benign.length) * 100 : 0;

  const stopPoints = detected.map((f) => f.stopPkts);
  stopPoints.sort((a, b) => a - b);
  let medianStopPointPct = 25;
  if (stopPoints.length > 0) {
    const medVal = stopPoints[Math.floor(stopPoints.length / 2)];
    medianStopPointPct = Math.round((medVal / 40) * 100);
  }

  const totalPktsWindow = flows.length * 40;
  const actualPktsUsed = flows.reduce((sum, f) => sum + f.stopPkts, 0);
  const observationSavedNum = totalPktsWindow > 0
    ? ((totalPktsWindow - actualPktsUsed) / totalPktsWindow) * 100
    : 0;

  const earlyAlerts = flows.filter((f) => f.status === 'Early alert').length;
  const unresolved = flows.filter((f) => f.status === 'No alert').length;

  return {
    coverage: coverageNum.toFixed(1),
    coverageDetail: `${detected.length}/${malicious.length} malicious detected*`,
    stopPoint: String(medianStopPointPct),
    stopPointDetail: `median stop across alerts*`,
    observationSaved: observationSavedNum.toFixed(1),
    observationDetail: `${totalPktsWindow - actualPktsUsed} pkts saved (${flows.length} flows)*`,
    fpr: fprNum.toFixed(2),
    fprDetail: `${falsePositives.length}/${benign.length} false alarms (≤1.0% target)*`,
    earlyAlertCount: earlyAlerts,
    unresolvedCount: unresolved,
    totalEvaluated: flows.length,
    maliciousCount: malicious.length,
    detectedMaliciousCount: detected.length,
    benignCount: benign.length,
    fpCount: falsePositives.length,
  };
}
