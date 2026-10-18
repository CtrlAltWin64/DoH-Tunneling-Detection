# Traceglass — Adaptive DoH Detection UI

A standalone React + TypeScript + Vite frontend concept for the Adaptive Stopping for Early Detection of Evasive DNS-over-HTTPS (DoH) Tunneling project.

## Isolation guarantee

This folder is designed to be copied into the repository as `frontend/`. It does not require or modify the Python research pipeline, tests, dataset files, root README, root requirements, or model code. All current screen values and flow rows are synthetic UI-preview data. No PCAPs are loaded, no model is called, and no research metrics should be presented as measured results.

## Run locally

Requires Node.js 20 or later.

```bash
npm install
npm run dev
```

Open the localhost URL printed by Vite. To run a production compile check:

```bash
npm run build
```

## Main screens

- **Overview:** synthetic KPIs, adaptive-versus-fixed progression chart, stopping-rule illustration, and recent flow decisions.
- **Flow explorer:** filter sample flows, select a row, and inspect a sample decision trace.
- **Evasion lab:** interactive low-and-slow / timing-jitter UI with sliders; changes only the illustration and does not execute a detector.
- **Experiment runs:** sample run cards and filters.

## Design tokens

The visual theme uses graphite and deep forest surfaces with acid-mint, cyber-teal, aqua-foam and cool-blue accents. The layout uses translucent glass panels, subtle borders, radial lighting, a responsive sidebar, and reduced-motion support.

## Before connecting real data

Connect the frontend to an agreed API contract only after the data preprocessing, model and evaluation interfaces are stable. Keep all real metrics sourced from the actual evaluation output. In particular, distinguish validation threshold selection from held-out test reporting, label simulated timing perturbations as simulations, and never treat flow-final features as early packet-prefix features.
