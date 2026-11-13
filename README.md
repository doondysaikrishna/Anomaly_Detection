# 🚀 AuraVision.AI - Autonomous Industrial Visual Anomaly Detection Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-6.0+-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, real-time visual inspection and deep defect localization platform built with **Self-Supervised PatchCore Memory Banks** and **Edge-Preserving Guided Bilateral Filtering**. 

Designed for mission-critical semiconductor, pharmaceutical, aerospace, and automotive assembly lines requiring **sub-100ms inference** and **pixel-precise contour segmentation** with zero defective training samples.

---

## 📸 Key Capabilities

1. **Self-Calibrated PatchCore Architecture**
   - High-dimensional ResNet-50 patch feature embeddings (`1536-D`).
   - Greedy coreset memory reduction for instant similarity lookup with minimal RAM footprint.
   - Zero defect training required — learns normal industrial structure patterns exclusively from healthy reference units.

2. **Real-Time Inspection Studio**
   - **Multi-Colormap Anomaly Heatmaps**: Turbo, Jet, Viridis, Inferno, Magma, Plasma.
   - **Guided Edge Refinement**: Fuses high-frequency image edge gradients with anomaly activation tensors for boundary-aligned defect contours.
   - **Interactive Split-Curtain Comparison**: Real-time slider comparing raw industrial image against localized defect heatmaps.
   - **Sub-10ms Dynamic Resegmentation**: Interactive threshold slider that updates bounding boxes, contours, and severity scoring instantly without re-running neural backbones.
   - **Micro-Defect Pixel Probe**: Hover over any pixel coordinate to inspect raw anomaly scores and defect probabilities.

3. **High-Throughput Batch QC Inspector**
   - Batch inspect entire production lots, conveyor snapshots, or folder structures.
   - Live yield monitoring, pass/fail conveyor rate, and automated CSV/JSON audit export.

4. **Multi-Object Dataset Explorer**
   - Built-in support for standard MVTec AD categories (Cable, Capsule, PCB / Circuit Board, Transistor, Pill, etc.).
   - Defect type taxonomy breakdown and 1-click model memory bank calibration.

5. **Deep Metrics & Telemetry Analytics**
   - Image & Pixel AUROC evaluation, F1-score optimization, and Confusion Matrices.
   - 57-sensor IoT telemetry streams, process unit health scores, and automated shift audit reports.

6. **Role-Based Authentication & User Profiles**
   - Lead QC Engineers, QA Auditors, Plant Managers, and Vision AI Specialists.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AuraVision React + Vite Frontend                    │
│                        http://localhost:5173                            │
│  [Inspection Studio]  [Batch QC]  [Object Explorer]  [Model Analytics]   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ JSON REST / Base64 Visuals
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Unified Python Backend Server                       │
│                        http://localhost:8000                            │
│   • 21+ Production REST Endpoints  • CORS Enabled  • Multi-User Auth    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│     PatchCore Vision Engine     │         │   Industrial IoT & Telemetry   │
│ • ResNet-50 Feature Embedder    │         │ • 57 Multi-Sensor Channels      │
│ • Coreset Memory Bank           │         │ • Process Unit Health (A-F)     │
│ • Guided Bilateral Refinement   │         │ • Shift Reports & Audit Logs    │
│ • Sub-10ms Resegmentation Cache │         │ • Automated Anomaly Alerts      │
└─────────────────────────────────┘         └─────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Prerequisites
- **Python 3.10+** with PyTorch and OpenCV.
- **Node.js 18+** & npm.

### 2. Launch All Services (One-Click)

On **Windows**:
```cmd
start.bat
```
or with Python:
```bash
python run_project.py
```

This will automatically launch:
- **Frontend Web UI:** `http://localhost:5173`
- **Backend API Server:** `http://localhost:8000`

---

## 🧪 Testing & Verification

Run the complete endpoint and model verification test suites:

### API Endpoints Test (16/16 Endpoints)
```bash
python test_dashboard.py
```

### PyTorch Neural Pipeline Test (All Categories)
```bash
python test_system.py
```

---

## 🔌 API Reference Highlights

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health, CUDA/CPU device status, active pipelines |
| `GET` | `/api/categories` | Discovered object categories, sample counts, defect types |
| `GET` | `/api/categories/{category}/samples` | Grouped train & test samples with ground truth masks |
| `POST` | `/api/inspect` | Full deep neural inspection with anomaly heatmap & bounding boxes |
| `POST` | `/api/inspect/resegment` | Sub-10ms dynamic re-thresholding without re-running neural backbone |
| `POST` | `/api/batch-inspect` | High-throughput batch processing for entire folders |
| `POST` | `/api/train` | 1-click model calibration & memory bank creation |
| `GET` | `/api/metrics` | Category AUROC, F1-scores, precision, recall, and confusion matrix |
| `GET` | `/api/reports` | Shift and daily quality audit reports with pass rate analytics |
| `POST` | `/api/auth/login` | Inspector login and session token generation |

---

## 📂 Project Directory Structure

```
Anomaly_Detection/
├── core/                           # Vision & ML Engine
│   ├── dataset/                    # Industrial Dataset Loaders
│   ├── evaluation/                 # AUROC & Metrics Benchmark
│   ├── models/                     # PatchCore Implementation
│   ├── vision/                     # Edge Filters & Heatmap Coloring
│   └── pipeline.py                 # Industrial Anomaly Pipeline
├── frontend/                       # React 18 + Vite + TypeScript UI
│   ├── src/
│   │   ├── components/             # Inspection Studio, Batch, Analytics, Navbar
│   │   ├── types/                  # Strict TypeScript Interfaces
│   │   ├── App.tsx                 # Root Component & Route Handler
│   │   └── main.tsx                # React Entrypoint
│   └── package.json
├── weights/                        # Calibrated PatchCore Memory Banks
├── test/                           # Industrial Dataset Samples (MVTec AD)
├── backend_server.py               # Unified High-Performance HTTP Server
├── run_project.py                  # Master Launch Orchestrator
├── start.bat                       # Windows 1-Click Launch Script
├── test_dashboard.py               # Complete REST API Test Suite
└── test_system.py                  # End-to-End PyTorch Test Suite
```

---

## 🛡️ License
Licensed under the [MIT License](LICENSE).
