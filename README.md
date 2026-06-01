# Automated Speech Emotion Recognition Monitoring Platform with Data Drift Detection

## Overview

A monitoring platform for Speech Emotion Recognition (SER) models that automatically detects data drift and model performance degradation in production environments.

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Research & Dataset Preparation | ✅ Done |
| 2 | Audio Preprocessing & Feature Extraction | 🔲 |
| 3 | Model Training & Evaluation | 🔲 |
| 4 | Data Drift Detection | 🔲 |
| 5 | Monitoring System & Automation | 🔲 |
| 6 | Dashboard & End-to-End Evaluation | 🔲 |

## Tech Stack

- **Dataset:** RAVDESS (cross-dataset drift simulation)
- **Features:** MFCC + Wav2Vec2 embeddings (`facebook/wav2vec2-base`)
- **Drift Detection:** KS Test + Wasserstein Distance (Evidently AI)
- **Automation:** n8n / Kestra
- **Dashboard:** Power BI (connected to CSV exports)

## Project Structure

```
├── data/
│   ├── raw/              # original RAVDESS audio files
│   ├── processed/        # resampled, normalized audio
│   ├── features/         # MFCC + Wav2Vec2 embeddings (.npy)
│   ├── drift_reports/    # drift detection outputs
│   └── monitoring_exports/  # CSV/XLSX for Power BI
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── drift_detection.py
│   └── monitoring.py
├── pipeline/             # n8n / Kestra workflow configs
├── models/               # saved model weights
├── configs/config.yaml
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```
