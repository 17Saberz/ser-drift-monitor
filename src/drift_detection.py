"""
Phase 4: Data Drift Detection using KS Test and Wasserstein Distance.

Compares feature distributions between reference (training) data and
incoming data to detect when the model's input distribution has shifted.

Drift scenarios:
  - baseline   : test split from same distribution as train (expect no drift)
  - noise      : Gaussian noise added to incoming features (simulate mic/env change)
  - speaker    : split by actor gender (train=male, incoming=female)

Output files:
  data/drift_reports/drift_report_{scenario}_{timestamp}.csv   per-feature stats
  data/monitoring_exports/drift_scores.csv                     aggregate scores (Power BI)
"""

import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from scipy import stats

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str) -> Path:
    """Return path as-is if it exists, otherwise resolve from project root."""
    p = Path(path)
    if p.exists():
        return p
    return PROJECT_ROOT / path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(_resolve(config_path), "r") as f:
        return yaml.safe_load(f)


# ── Drift Detector ────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Fits on a reference distribution (training features) and detects
    drift in incoming data using KS Test and Wasserstein Distance.
    """

    def __init__(self, ks_threshold: float = 0.05, wasserstein_threshold: float = 0.1):
        self.ks_threshold = ks_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.reference = None
        self.n_features = None

    def fit(self, X_reference: np.ndarray) -> "DriftDetector":
        """Store reference feature matrix (N_ref x F)."""
        self.reference = X_reference
        self.n_features = X_reference.shape[1]
        return self

    def detect(self, X_incoming: np.ndarray, feature_names: list = None) -> pd.DataFrame:
        """
        Run per-feature KS test and Wasserstein distance.
        Returns a DataFrame with one row per feature.
        """
        assert self.reference is not None, "Call fit() first."
        assert X_incoming.shape[1] == self.n_features

        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(self.n_features)]

        rows = []
        for i in range(self.n_features):
            ref_col = self.reference[:, i]
            inc_col = X_incoming[:, i]

            ks_stat, ks_pval = stats.ks_2samp(ref_col, inc_col)
            w_dist = stats.wasserstein_distance(ref_col, inc_col)

            rows.append({
                "feature":            feature_names[i],
                "ks_statistic":       round(ks_stat, 6),
                "ks_pvalue":          round(ks_pval, 6),
                "ks_drift":           ks_pval < self.ks_threshold,
                "wasserstein_dist":   round(w_dist, 6),
                "wasserstein_drift":  w_dist > self.wasserstein_threshold,
                "ref_mean":           round(ref_col.mean(), 4),
                "inc_mean":           round(inc_col.mean(), 4),
                "mean_shift":         round(abs(inc_col.mean() - ref_col.mean()), 4),
            })

        return pd.DataFrame(rows)

    def summary(self, report: pd.DataFrame) -> dict:
        """Aggregate drift report into a single score dict."""
        n = len(report)
        ks_drifted   = report["ks_drift"].sum()
        w_drifted    = report["wasserstein_drift"].sum()

        return {
            "n_features":           n,
            "ks_drifted_features":  int(ks_drifted),
            "ks_drift_ratio":       round(ks_drifted / n, 4),
            "w_drifted_features":   int(w_drifted),
            "w_drift_ratio":        round(w_drifted / n, 4),
            "mean_ks_statistic":    round(report["ks_statistic"].mean(), 4),
            "mean_wasserstein":     round(report["wasserstein_dist"].mean(), 4),
            "max_mean_shift":       round(report["mean_shift"].max(), 4),
        }


# ── Drift Scenarios ───────────────────────────────────────────────────────────

def scenario_baseline(X: np.ndarray, metadata: pd.DataFrame):
    """No drift: train split vs test split (same distribution)."""
    train_mask = metadata["split"] == "train"
    test_mask  = metadata["split"] == "test"
    return X[train_mask.values], X[test_mask.values], metadata[test_mask].reset_index(drop=True)


def scenario_noise(X: np.ndarray, metadata: pd.DataFrame,
                   noise_std: float = 1.0, random_state: int = 42):
    """
    Noise injection: add Gaussian noise to test features to simulate
    microphone change or background noise in production.
    """
    rng = np.random.default_rng(random_state)
    train_mask = metadata["split"] == "train"
    test_mask  = metadata["split"] == "test"

    X_ref = X[train_mask.values]
    X_inc = X[test_mask.values] + rng.normal(0, noise_std, X[test_mask.values].shape)
    return X_ref, X_inc, metadata[test_mask].reset_index(drop=True)


def scenario_speaker_split(X: np.ndarray, metadata: pd.DataFrame):
    """
    Speaker split: reference = male actors, incoming = female actors.
    Simulates demographic shift (speaker variability drift).
    """
    male_mask   = metadata["gender"] == "male"
    female_mask = metadata["gender"] == "female"
    return (
        X[male_mask.values],
        X[female_mask.values],
        metadata[female_mask].reset_index(drop=True),
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def run_drift_detection(
    features_dir:  str = "data/features",
    drift_dir:     str = "data/drift_reports",
    exports_dir:   str = "data/monitoring_exports",
    config_path:   str = "configs/config.yaml",
    noise_std:     float = 1.0,
) -> dict:

    cfg = load_config(config_path)
    dc  = cfg["drift_detection"]
    _resolve(drift_dir).mkdir(parents=True, exist_ok=True)
    _resolve(exports_dir).mkdir(parents=True, exist_ok=True)

    # Load features
    feat_path = _resolve(features_dir)
    X = np.load(feat_path / "mfcc_features.npy")           # (N, 240)
    metadata  = pd.read_csv(feat_path / "metadata.csv")
    feature_names = [f"mfcc_{i}" for i in range(X.shape[1])]

    detector = DriftDetector(
        ks_threshold=dc["ks_test_threshold"],
        wasserstein_threshold=dc["wasserstein_threshold"],
    )

    scenarios = {
        "baseline":     lambda: scenario_baseline(X, metadata),
        "noise":        lambda: scenario_noise(X, metadata, noise_std=noise_std),
        "speaker_split": lambda: scenario_speaker_split(X, metadata),
    }

    all_summaries = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for name, fn in scenarios.items():
        print(f"\n{'─'*50}")
        print(f"Scenario: {name}")
        X_ref, X_inc, inc_meta = fn()
        print(f"  Reference : {X_ref.shape}   Incoming: {X_inc.shape}")

        detector.fit(X_ref)
        report = detector.detect(X_inc, feature_names=feature_names)
        summ   = detector.summary(report)
        summ["scenario"]  = name
        summ["timestamp"] = timestamp
        summ["noise_std"] = noise_std if name == "noise" else None

        print(f"  KS drift    : {summ['ks_drifted_features']}/{summ['n_features']} "
              f"features ({summ['ks_drift_ratio']:.1%})")
        print(f"  Wasserstein : {summ['w_drifted_features']}/{summ['n_features']} "
              f"features ({summ['w_drift_ratio']:.1%})")
        print(f"  Mean KS stat: {summ['mean_ks_statistic']}   "
              f"Mean W-dist: {summ['mean_wasserstein']}")

        # Save per-feature report
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = _resolve(drift_dir) / f"drift_report_{name}_{ts_file}.csv"
        report.to_csv(report_path, index=False)

        all_summaries.append(summ)

    # Append aggregate scores to monitoring_exports (Power BI trend)
    scores_df   = pd.DataFrame(all_summaries)
    scores_file = _resolve(exports_dir) / "drift_scores.csv"
    scores_df.to_csv(
        scores_file,
        mode="a",
        header=not scores_file.exists(),
        index=False,
    )
    print(f"\nDrift scores saved to {scores_file}")

    return {s["scenario"]: s for s in all_summaries}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run drift detection scenarios")
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--drift_dir",    default="data/drift_reports")
    parser.add_argument("--exports_dir",  default="data/monitoring_exports")
    parser.add_argument("--noise_std",    type=float, default=1.0,
                        help="Std of Gaussian noise for noise scenario")
    args = parser.parse_args()

    run_drift_detection(
        features_dir=args.features_dir,
        drift_dir=args.drift_dir,
        exports_dir=args.exports_dir,
        noise_std=args.noise_std,
    )
