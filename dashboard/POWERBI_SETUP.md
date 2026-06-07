# Power BI Dashboard Setup Guide

## Step 1 — Generate Power BI tables

Run this from the project root every time you want to refresh the data:
```bash
python src/export_powerbi.py
```

This creates 6 clean CSV files in `data/monitoring_exports/`:

| File | Contents | Key Visuals |
|---|---|---|
| `pb_monitoring_log.csv` | Time-series: F1, drift, alerts per run | Line charts, KPI cards |
| `pb_alert_summary.csv` | Alert counts per scenario | Bar chart, risk table |
| `pb_model_baseline.csv` | Phase 3 baseline accuracy + F1 | KPI reference cards |
| `pb_class_performance.csv` | Per-emotion precision, recall, F1 | Bar chart, table |
| `pb_drift_comparison.csv` | KS + Wasserstein per scenario | Grouped bar chart |
| `pb_confusion_matrix.csv` | Long-format confusion matrix | Matrix visual |

---

## Step 2 — Connect Power BI to CSV files

1. Open **Power BI Desktop**
2. Click **Home → Get Data → Text/CSV**
3. Load each `pb_*.csv` file from `data/monitoring_exports/`
4. Repeat for all 6 files
5. Click **Close & Apply**

---

## Step 3 — Set up Auto-Refresh

1. In Power BI Desktop → **Home → Transform Data**
2. For each query, set the file path as a **parameter** so refreshing works:
   - Right-click the query → **Advanced Editor**
   - The file path is already set — just click **Refresh** to reload

---

## Step 4 — Build the Dashboard (4 pages)

### Page 1 — Model Performance Overview
| Visual | Data | Fields |
|---|---|---|
| KPI Card — Baseline F1 | pb_model_baseline | test_f1 |
| KPI Card — Latest F1 | pb_monitoring_log | batch_f1 (last row) |
| KPI Card — Latest Drift | pb_monitoring_log | ks_drift_ratio (last row) |
| Bar chart — Per-class F1 | pb_class_performance | emotion vs f1_score |
| Table — Classification Report | pb_class_performance | all columns |

### Page 2 — Monitoring Time Series
| Visual | Data | Fields |
|---|---|---|
| Line chart — F1 over time | pb_monitoring_log | timestamp vs batch_f1 |
| Line chart — Drift over time | pb_monitoring_log | timestamp vs ks_drift_ratio |
| Slicer — Scenario filter | pb_monitoring_log | scenario |
| Alert flag line | pb_monitoring_log | alert_flag (as dotted overlay) |

### Page 3 — Drift Analysis
| Visual | Data | Fields |
|---|---|---|
| Clustered bar — KS drift | pb_drift_comparison | scenario vs ks_drift_ratio |
| Clustered bar — Wasserstein | pb_drift_comparison | scenario vs mean_wasserstein |
| Table — Drift summary | pb_drift_comparison | all columns |
| Matrix — Confusion matrix | pb_confusion_matrix | true_emotion vs predicted_emotion, count |

### Page 4 — Alert Dashboard
| Visual | Data | Fields |
|---|---|---|
| Card — Total alerts | pb_alert_summary | total_alerts (sum) |
| Bar chart — Alert rate | pb_alert_summary | scenario vs alert_rate |
| Table — Alert log | pb_monitoring_log | timestamp, scenario, status |
| Donut — Healthy vs Unhealthy | pb_monitoring_log | health_status |

---

## Step 5 — Refresh after each monitoring run

Each time you run the monitoring pipeline:
```bash
python src/export_powerbi.py    # refresh CSV exports
```
Then in Power BI: **Home → Refresh** to update all visuals.

---

## Color Theme (matches project)

| Scenario / Status | Hex Color |
|---|---|
| Baseline / Healthy | `#27ae60` (green) |
| Noise / Alert | `#e74c3c` (red) |
| Speaker Split | `#f39c12` (orange) |
| Gradual | `#9b59b6` (purple) |
| Neutral | `#3498db` (blue) |
