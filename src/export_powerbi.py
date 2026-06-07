"""
Phase 6: Consolidate all monitoring outputs into clean Power BI-ready tables.

Run this script after any monitoring session to refresh the Power BI data.
Power BI connects to these files via "Get Data > Text/CSV" with auto-refresh.

Output files (all in data/monitoring_exports/):
  pb_monitoring_log.csv       main time-series table (F1, drift, alerts)
  pb_alert_summary.csv        alert counts and rates per scenario
  pb_model_baseline.csv       baseline model performance (from Phase 3)
  pb_class_performance.csv    per-emotion F1, precision, recall
  pb_drift_comparison.csv     KS + Wasserstein across scenarios
  pb_confusion_matrix.csv     confusion matrix (long format for Power BI)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR  = PROJECT_ROOT / "data" / "monitoring_exports"


def _read(filename: str) -> pd.DataFrame | None:
    path = EXPORTS_DIR / filename
    if not path.exists():
        print(f"  [skip] {filename} not found")
        return None
    return pd.read_csv(path)


def export_monitoring_log():
    """Main time-series table — one row per monitoring run."""
    df = _read("monitoring_log.csv")
    if df is None:
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date
    df["hour"]      = df["timestamp"].dt.hour
    df["alert_flag"] = df["drift_alert"] | df["perf_alert"]
    df["health_status"] = df["alert_flag"].map({True: "Unhealthy", False: "Healthy"})
    df.to_csv(EXPORTS_DIR / "pb_monitoring_log.csv", index=False)
    print(f"  pb_monitoring_log.csv       {len(df)} rows")


def export_alert_summary():
    """Alert count and rate per scenario."""
    df = _read("alert_log.csv")
    if df is None:
        return
    summary = df.groupby("scenario").agg(
        total_runs     = ("timestamp", "count"),
        total_alerts   = ("any_alert", "sum"),
        avg_f1         = ("f1_score", "mean"),
        min_f1         = ("f1_score", "min"),
        avg_drift      = ("ks_drift_ratio", "mean"),
        max_drift      = ("ks_drift_ratio", "max"),
    ).reset_index().round(4)
    summary["alert_rate"] = (summary["total_alerts"] / summary["total_runs"]).round(4)
    summary["risk_level"] = summary["alert_rate"].apply(
        lambda x: "High" if x > 0.6 else ("Medium" if x > 0.2 else "Low")
    )
    summary.to_csv(EXPORTS_DIR / "pb_alert_summary.csv", index=False)
    print(f"  pb_alert_summary.csv        {len(summary)} rows")


def export_model_baseline():
    """Baseline model performance from Phase 3 training."""
    path = EXPORTS_DIR / "model_metrics.csv"
    if not path.exists():
        print(f"  [skip] model_metrics.csv not found")
        return
    # Use on_bad_lines='skip' to ignore 13-col monitoring rows mixed in from old runs
    df = pd.read_csv(path, on_bad_lines="skip")
    if df.empty:
        print(f"  [skip] model_metrics.csv has no valid rows")
        return
    # Keep only rows that have the Phase 3 columns (test_f1 must exist)
    if "test_f1" not in df.columns:
        print(f"  [skip] model_metrics.csv missing expected columns")
        return
    df = df.dropna(subset=["test_f1"])
    if df.empty:
        print(f"  [skip] model_metrics.csv has no rows with test_f1")
        return
    latest = df.iloc[[-1]].copy()
    latest["phase"] = "Phase 3 — Baseline Training"
    latest.to_csv(EXPORTS_DIR / "pb_model_baseline.csv", index=False)
    print(f"  pb_model_baseline.csv       1 row")


def export_class_performance():
    """Per-emotion classification metrics."""
    df = _read("classification_report.csv")
    if df is None:
        return
    emotions = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised"
    ]
    df_emo = df[df["label"].isin(emotions)].copy()
    df_emo = df_emo[["label", "precision", "recall", "f1_score", "support"]].round(4)
    df_emo.columns = ["emotion", "precision", "recall", "f1_score", "support"]
    df_emo["f1_category"] = df_emo["f1_score"].apply(
        lambda x: "Good (>0.7)" if x > 0.7 else ("Fair (0.4–0.7)" if x > 0.4 else "Poor (<0.4)")
    )
    df_emo.to_csv(EXPORTS_DIR / "pb_class_performance.csv", index=False)
    print(f"  pb_class_performance.csv    {len(df_emo)} rows")


def export_drift_comparison():
    """Drift scores across all scenarios."""
    df = _read("drift_scores.csv")
    if df is None:
        return
    latest = df.drop_duplicates("scenario", keep="last").copy()
    latest["drift_level"] = latest["ks_drift_ratio"].apply(
        lambda x: "High (>50%)" if x > 0.5 else ("Medium (10–50%)" if x > 0.1 else "Low (<10%)")
    )
    latest["performance_impact"] = ["Low", "High", "None"][:len(latest)]
    latest.to_csv(EXPORTS_DIR / "pb_drift_comparison.csv", index=False)
    print(f"  pb_drift_comparison.csv     {len(latest)} rows")


def export_confusion_matrix():
    """Convert confusion matrix to long format (required for Power BI matrix visual)."""
    df = _read("confusion_matrix.csv")
    if df is None:
        return
    df = df.rename_axis("true_emotion").reset_index()
    long = df.melt(id_vars="true_emotion", var_name="predicted_emotion", value_name="count")
    long["correct"] = long["true_emotion"] == long["predicted_emotion"]
    long.to_csv(EXPORTS_DIR / "pb_confusion_matrix.csv", index=False)
    print(f"  pb_confusion_matrix.csv     {len(long)} rows")


def run_export():
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting Power BI tables → {EXPORTS_DIR}")
    print("─" * 50)
    export_monitoring_log()
    export_alert_summary()
    export_model_baseline()
    export_class_performance()
    export_drift_comparison()
    export_confusion_matrix()
    print("─" * 50)
    print(f"Done. Open Power BI Desktop and refresh data sources.")
    print(f"See dashboard/POWERBI_SETUP.md for connection guide.\n")


if __name__ == "__main__":
    run_export()
