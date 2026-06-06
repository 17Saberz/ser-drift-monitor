"""
Phase 5: Automated Monitoring Pipeline for SER with Data Drift Detection.

Runs as a scheduled job (via n8n, Kestra, or Python scheduler).
Each run:
  1. Simulates an incoming batch of audio feature vectors
  2. Runs model inference
  3. Runs drift detection vs reference distribution
  4. Checks performance drop vs baseline
  5. Fires alert if thresholds exceeded
  6. Exports results to monitoring_exports/ for Power BI

Usage:
  python src/monitoring.py                    # single run
  python src/monitoring.py --schedule 60      # run every 60 seconds
  python src/monitoring.py --scenario noise   # simulate drift scenario
"""

import json
import joblib
import argparse
import time
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.exists() else PROJECT_ROOT / path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(_resolve(config_path)) as f:
        return yaml.safe_load(f)


# ── Alert System ─────────────────────────────────────────────────────────────

class AlertSystem:
    """Handles drift and performance alerts."""

    def __init__(self, exports_dir: str = "data/monitoring_exports"):
        self.exports_dir = _resolve(exports_dir)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.alert_log = self.exports_dir / "alert_log.csv"

    def check_and_fire(
        self,
        scenario: str,
        ks_drift_ratio: float,
        f1_score: float,
        baseline_f1: float,
        drift_threshold: float = 0.3,
        f1_drop_threshold: float = 0.05,
    ) -> dict:
        """
        Evaluate alert conditions and log result.
        Returns alert dict with triggered flags.
        """
        f1_drop    = baseline_f1 - f1_score
        drift_alert = ks_drift_ratio > drift_threshold
        perf_alert  = f1_drop > f1_drop_threshold
        any_alert   = drift_alert or perf_alert

        alert = {
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario":        scenario,
            "ks_drift_ratio":  round(ks_drift_ratio, 4),
            "f1_score":        round(f1_score, 4),
            "baseline_f1":     round(baseline_f1, 4),
            "f1_drop":         round(f1_drop, 4),
            "drift_alert":     drift_alert,
            "perf_alert":      perf_alert,
            "any_alert":       any_alert,
            "status":          "🚨 ALERT" if any_alert else "✅ OK",
        }

        # Console output
        print(f"\n{'='*55}")
        print(f"  MONITORING ALERT CHECK — {alert['timestamp']}")
        print(f"{'='*55}")
        print(f"  Scenario      : {scenario}")
        print(f"  KS Drift      : {ks_drift_ratio:.1%}  "
              f"{'🚨' if drift_alert else '✅'} (threshold: {drift_threshold:.0%})")
        print(f"  F1 Score      : {f1_score:.4f}  "
              f"(baseline: {baseline_f1:.4f}, drop: {f1_drop:+.4f})")
        print(f"  Perf Alert    : {'🚨 YES — F1 dropped' if perf_alert else '✅ NO'}")
        print(f"  Overall       : {alert['status']}")
        print(f"{'='*55}")

        # Append to alert log (Power BI)
        row = pd.DataFrame([alert])
        row.to_csv(
            self.alert_log,
            mode="a",
            header=not self.alert_log.exists(),
            index=False,
        )

        return alert


# ── Batch Simulator ──────────────────────────────────────────────────────────

def simulate_batch(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    scenario: str = "baseline",
    batch_size: int = 50,
    noise_std: float = 1.0,
    random_state: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate an incoming batch of feature vectors.

    Scenarios:
      baseline     — random sample from test split (no drift)
      noise        — test sample + Gaussian noise
      speaker_split — female speakers only
      gradual      — increasing noise level over time (drift creep)
    """
    rng = np.random.default_rng(random_state or int(time.time()))

    if scenario == "speaker_split":
        mask = metadata["gender"] == "female"
        pool_X = X[mask.values]
        pool_y = y[mask.values]
    else:
        mask = metadata["split"] == "test"
        pool_X = X[mask.values]
        pool_y = y[mask.values]

    idx = rng.choice(len(pool_X), size=min(batch_size, len(pool_X)), replace=False)
    X_batch = pool_X[idx].copy()
    y_batch = pool_y[idx]

    if scenario == "noise":
        X_batch += rng.normal(0, noise_std, X_batch.shape)
    elif scenario == "gradual":
        # Gradually increasing noise — simulate slow environment degradation
        hour = datetime.now().hour
        drift_factor = hour / 24.0
        X_batch += rng.normal(0, noise_std * drift_factor, X_batch.shape)

    return X_batch, y_batch


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_monitoring_pipeline(
    config_path:   str = "configs/config.yaml",
    features_dir:  str = "data/features",
    models_dir:    str = "models",
    exports_dir:   str = "data/monitoring_exports",
    scenario:      str = "baseline",
    batch_size:    int = 50,
    noise_std:     float = 1.0,
) -> dict:

    cfg = load_config(config_path)
    dc  = cfg["drift_detection"]
    mc  = cfg["monitoring"]

    # ── Load resources ────────────────────────────────────────────────────────
    feat_path = _resolve(features_dir)
    X         = np.load(feat_path / "mfcc_features.npy")
    y         = np.load(feat_path / "labels.npy")
    metadata  = pd.read_csv(feat_path / "metadata.csv")

    with open(feat_path / "label_map.json") as f:
        label_map = json.load(f)
    idx2label = {v: k for k, v in label_map.items()}

    model  = joblib.load(_resolve(models_dir) / "mlp_model.joblib")
    scaler = joblib.load(_resolve(models_dir) / "scaler.joblib")

    # ── Reference distribution (train split) ──────────────────────────────────
    train_mask = metadata["split"] == "train"
    X_ref      = X[train_mask.values]

    # ── Baseline F1 (from Phase 3 training output) ───────────────────────────
    # model_metrics.csv is written by src/model.py with a fixed 10-column schema.
    # We only read from it here — monitoring writes to monitoring_log.csv instead.
    train_metrics_file = _resolve(exports_dir) / "model_metrics.csv"
    if train_metrics_file.exists():
        try:
            baseline_f1 = pd.read_csv(train_metrics_file)["test_f1"].iloc[-1]
        except Exception:
            baseline_f1 = 0.5986
    else:
        baseline_f1 = 0.5986  # fallback from Phase 3 results

    # ── Simulate incoming batch ───────────────────────────────────────────────
    print(f"\nSimulating incoming batch — scenario: '{scenario}' | size: {batch_size}")
    X_batch, y_batch = simulate_batch(
        X, y, metadata,
        scenario=scenario,
        batch_size=batch_size,
        noise_std=noise_std,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    X_batch_s = scaler.transform(X_batch)
    y_pred    = model.predict(X_batch_s)
    batch_f1  = f1_score(y_batch, y_pred, average="macro", zero_division=0)
    batch_acc = accuracy_score(y_batch, y_pred)
    print(f"  Inference done — Acc: {batch_acc:.4f}  F1: {batch_f1:.4f}")

    # ── Drift Detection ───────────────────────────────────────────────────────
    try:
        from src.drift_detection import DriftDetector   # called as module (notebooks/api)
    except ModuleNotFoundError:
        from drift_detection import DriftDetector       # called directly (terminal)

    detector = DriftDetector(
        ks_threshold=dc["ks_test_threshold"],
        wasserstein_threshold=dc["wasserstein_threshold"],
    )
    detector.fit(X_ref)
    drift_report = detector.detect(X_batch)
    drift_summary = detector.summary(drift_report)
    ks_drift_ratio = drift_summary["ks_drift_ratio"]

    # ── Alert Check ───────────────────────────────────────────────────────────
    alerter = AlertSystem(exports_dir)
    alert = alerter.check_and_fire(
        scenario=scenario,
        ks_drift_ratio=ks_drift_ratio,
        f1_score=batch_f1,
        baseline_f1=baseline_f1,
        drift_threshold=mc["alert_threshold"]["drift_score"],
        f1_drop_threshold=mc["alert_threshold"]["f1_drop"],
    )

    # ── Export monitoring report (Power BI) ───────────────────────────────────
    exports = _resolve(exports_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_row = pd.DataFrame([{
        "timestamp":        timestamp,
        "scenario":         scenario,
        "batch_size":       batch_size,
        "batch_acc":        round(batch_acc,  4),
        "batch_f1":         round(batch_f1,   4),
        "baseline_f1":      round(baseline_f1, 4),
        "f1_drop":          round(baseline_f1 - batch_f1, 4),
        "ks_drift_ratio":   round(ks_drift_ratio, 4),
        "mean_ks_stat":     drift_summary["mean_ks_statistic"],
        "mean_wasserstein": drift_summary["mean_wasserstein"],
        "drift_alert":      alert["drift_alert"],
        "perf_alert":       alert["perf_alert"],
        "status":           alert["status"],
    }])

    # Write to monitoring_log.csv (separate from model_metrics.csv)
    monitor_file = exports / "monitoring_log.csv"
    report_row.to_csv(
        monitor_file,
        mode="a",
        header=not monitor_file.exists(),
        index=False,
    )

    # Save per-batch predictions
    label_names  = [idx2label[i] for i in y_batch]
    pred_names   = [idx2label[i] for i in y_pred]
    preds_df = pd.DataFrame({
        "timestamp": timestamp,
        "scenario":  scenario,
        "true":      label_names,
        "predicted": pred_names,
        "correct":   [t == p for t, p in zip(label_names, pred_names)],
    })
    preds_file = exports / "batch_predictions.csv"
    preds_df.to_csv(
        preds_file,
        mode="a",
        header=not preds_file.exists(),
        index=False,
    )

    print(f"  Report saved → {monitor_file.name}")
    return alert


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_scheduler(interval_seconds: int, scenario: str, **kwargs):
    """Run monitoring pipeline on a fixed interval."""
    print(f"Scheduler started — interval: {interval_seconds}s | scenario: {scenario}")
    print("Press Ctrl+C to stop.\n")
    run_count = 0
    while True:
        run_count += 1
        print(f"\n[Run #{run_count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            run_monitoring_pipeline(scenario=scenario, **kwargs)
        except Exception as e:
            print(f"  ERROR: {e}")
        print(f"  Next run in {interval_seconds}s ...")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SER Monitoring Pipeline")
    parser.add_argument("--scenario", default="baseline",
                        choices=["baseline", "noise", "speaker_split", "gradual"])
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--noise_std",  type=float, default=1.0)
    parser.add_argument("--schedule",   type=int, default=0,
                        help="Repeat every N seconds (0 = run once)")
    args = parser.parse_args()

    kwargs = dict(scenario=args.scenario, batch_size=args.batch_size,
                  noise_std=args.noise_std)

    if args.schedule > 0:
        run_scheduler(args.schedule, **kwargs)
    else:
        run_monitoring_pipeline(**kwargs)
