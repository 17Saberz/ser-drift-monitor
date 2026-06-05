"""
Phase 3: Model training and evaluation for Speech Emotion Recognition.

Uses scikit-learn MLPClassifier trained on MFCC features.
Exports metrics and predictions to data/monitoring_exports/ for Power BI.

Saved files:
  models/mlp_model.joblib          trained model
  models/scaler.joblib             fitted StandardScaler
  data/monitoring_exports/
    model_metrics.csv              accuracy, F1 per run (append-mode for tracking)
    classification_report.csv      per-class precision, recall, F1
    confusion_matrix.csv           raw confusion matrix
    predictions.csv                file-level predictions (for drift analysis)
"""

import json
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.exists() else PROJECT_ROOT / path


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(_resolve(config_path), "r") as f:
        return yaml.safe_load(f)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(features_dir: str = "data/features", feature_type: str = "mfcc"):
    """
    Load features, labels, metadata and label map.
    feature_type: 'mfcc' (240-dim) or 'wav2vec2' (768-dim)
    Returns X_train, X_test, y_train, y_test, metadata, idx2label
    """
    feat_path = _resolve(features_dir)

    if feature_type == "wav2vec2" and (feat_path / "wav2vec2_features.npy").exists():
        X = np.load(feat_path / "wav2vec2_features.npy")
        print(f"Using Wav2Vec2 features: {X.shape}")
    else:
        X = np.load(feat_path / "mfcc_features.npy")
        print(f"Using MFCC features: {X.shape}")

    y = np.load(feat_path / "labels.npy")
    metadata = pd.read_csv(feat_path / "metadata.csv")

    with open(feat_path / "label_map.json") as f:
        label_map = json.load(f)
    idx2label = {v: k for k, v in label_map.items()}

    train_mask = metadata["split"] == "train"
    test_mask  = metadata["split"] == "test"

    X_train, y_train = X[train_mask.values], y[train_mask.values]
    X_test,  y_test  = X[test_mask.values],  y[test_mask.values]

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, metadata, idx2label


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> MLPClassifier:
    mc = cfg["model"]
    # Note: sklearn MLPClassifier has no dropout param (that's PyTorch-only).
    # Regularization is handled via alpha (L2 penalty) instead.
    return MLPClassifier(
        hidden_layer_sizes=tuple(mc["hidden_dims"]),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=mc["batch_size"],
        learning_rate_init=mc["learning_rate"],
        max_iter=mc["epochs"],
        random_state=mc["random_state"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    config_path: str = "configs/config.yaml",
    features_dir: str = "data/features",
    models_dir: str = "models",
    exports_dir: str = "data/monitoring_exports",
    feature_type: str = "mfcc",
    run_cv: bool = True,
) -> dict:

    cfg = load_config(config_path)
    _resolve(models_dir).mkdir(parents=True, exist_ok=True)
    _resolve(exports_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, metadata, idx2label = load_features(
        features_dir, feature_type
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Cross-validation ──────────────────────────────────────────────────────
    if run_cv:
        print("\nRunning 5-fold cross-validation on train set ...")
        model_cv = build_model(cfg)
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=cfg["model"]["random_state"])
        cv_scores = cross_val_score(model_cv, X_train_s, y_train,
                                    cv=cv, scoring="f1_macro", n_jobs=-1)
        print(f"  CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    else:
        cv_scores = np.array([])

    # ── Final training ────────────────────────────────────────────────────────
    print("\nTraining final model ...")
    model = build_model(cfg)
    # Remove None kwargs (dropout not supported in sklearn MLP)
    model.fit(X_train_s, y_train)
    print(f"  Iterations: {model.n_iter_}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred_train = model.predict(X_train_s)
    y_pred_test  = model.predict(X_test_s)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred_test)
    train_f1  = f1_score(y_train, y_pred_train, average="macro")
    test_f1   = f1_score(y_test,  y_pred_test,  average="macro")

    label_names = [idx2label[i] for i in sorted(idx2label)]
    report_dict = classification_report(
        y_test, y_pred_test, target_names=label_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred_test)

    print(f"\n{'='*45}")
    print(f"  Train Accuracy : {train_acc:.4f}   F1-macro: {train_f1:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}   F1-macro: {test_f1:.4f}")
    if cv_scores.size:
        print(f"  CV F1-macro    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"{'='*45}")

    # ── Save model & scaler ───────────────────────────────────────────────────
    joblib.dump(model,  _resolve(models_dir) / "mlp_model.joblib")
    joblib.dump(scaler, _resolve(models_dir) / "scaler.joblib")
    print(f"\nModel saved to {models_dir}/")

    # ── Export metrics ────────────────────────────────────────────────────────
    exports = _resolve(exports_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Append-mode metrics log (for Power BI trend line)
    metrics_row = pd.DataFrame([{
        "timestamp":   timestamp,
        "feature_type": feature_type,
        "train_acc":   round(train_acc, 4),
        "test_acc":    round(test_acc,  4),
        "train_f1":    round(train_f1,  4),
        "test_f1":     round(test_f1,   4),
        "cv_f1_mean":  round(cv_scores.mean(), 4) if cv_scores.size else None,
        "cv_f1_std":   round(cv_scores.std(),  4) if cv_scores.size else None,
        "n_train":     len(X_train),
        "n_test":      len(X_test),
    }])
    metrics_file = exports / "model_metrics.csv"
    metrics_row.to_csv(
        metrics_file,
        mode="a",
        header=not metrics_file.exists(),
        index=False,
    )

    # Per-class report
    report_df = pd.DataFrame(report_dict).T.reset_index()
    report_df.columns = ["label", "precision", "recall", "f1_score", "support"]
    report_df.insert(0, "timestamp", timestamp)
    report_df.to_csv(exports / "classification_report.csv", index=False)

    # Confusion matrix
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(exports / "confusion_matrix.csv")

    # File-level predictions (path, true label, predicted label, correct)
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
    preds_df = test_meta[["path", "emotion", "actor", "gender"]].copy()
    preds_df["predicted"]  = [idx2label[i] for i in y_pred_test]
    preds_df["correct"]    = preds_df["emotion"] == preds_df["predicted"]
    preds_df["timestamp"]  = timestamp
    preds_df.to_csv(exports / "predictions.csv", index=False)

    print(f"Metrics exported to {exports_dir}/")

    return {
        "train_acc": train_acc, "test_acc": test_acc,
        "train_f1": train_f1,   "test_f1": test_f1,
        "cv_scores": cv_scores,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SER emotion classifier")
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--feature_type", choices=["mfcc", "wav2vec2"], default="mfcc")
    parser.add_argument("--models_dir",   default="models")
    parser.add_argument("--exports_dir",  default="data/monitoring_exports")
    parser.add_argument("--no_cv", action="store_true", help="Skip cross-validation")
    args = parser.parse_args()

    train(
        features_dir=args.features_dir,
        feature_type=args.feature_type,
        models_dir=args.models_dir,
        exports_dir=args.exports_dir,
        run_cv=not args.no_cv,
    )
