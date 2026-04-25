# Nvex Physical AI Demo

A 7-page interactive web demo showcasing Nvex as the **agent-in-the-loop orchestration layer** for Physical AI post-training.

## Pages

| Route | Page |
|-------|------|
| Home | Project Hub — intelligence loop diagram, platform metrics, project list |
| Overview | Project Overview — KPI cards, task breakdown, loop position, next action |
| Failure Map | Interactive failure clusters, radar chart, root-cause diagnosis |
| Patch Plan | Data targeting, training strategy, verification, expected uplift |
| Iteration Runner | Animated timeline, live console, artifact tracker |
| Improvement Report | Before/after metrics, assets created, next iteration suggestion |
| Platform Memory | Recipes, pipeline templates, failure ontology, compounding chart |

## Quick Start

```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # production build → dist/
```

## Stack

- React 19 + Vite 8
- Pure CSS (no UI framework) — dark `#07090f` theme, indigo-violet gradients
- SVG for Intelligence Loop diagram and radar chart
- All data mocked in `src/data/mockData.js`

## Story

Demo follows the LIBERO Kitchen Pick-and-Place scenario:
`NeuroVLA-LIBERO-ckpt_v0.7` at 62% success → Nvex diagnoses failure clusters →
generates patch plan → AlphaBrain CL update → `ckpt_v0.8` at 74% (+12%).
