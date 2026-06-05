"""
Feature extraction: MFCC (40 + delta + delta-delta) and Wav2Vec2 embeddings.
Saves features to data/features/ as .npy files with a metadata CSV.

Output files:
  data/features/mfcc_features.npy       shape (N, 240)
  data/features/wav2vec2_features.npy   shape (N, 768)
  data/features/labels.npy              shape (N,)   int-encoded
  data/features/metadata.csv            path, emotion, actor, gender, split
  data/features/label_map.json          emotion -> int mapping
"""

import json
import numpy as np
import pandas as pd
import librosa
import yaml
from pathlib import Path
from tqdm import tqdm

# torch and transformers are imported lazily inside Wav2Vec2Extractor
# so that --no_wav2vec2 runs without requiring a working PyTorch install.

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.exists() else PROJECT_ROOT / path


EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised",
]
LABEL2IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(_resolve(config_path), "r") as f:
        return yaml.safe_load(f)


# ── MFCC ─────────────────────────────────────────────────────────────────────

def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Returns a (3 * n_mfcc * 2,) = 240-dim vector:
    [mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std]
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1),
    ])  # (240,)


# ── Wav2Vec2 ──────────────────────────────────────────────────────────────────

class Wav2Vec2Extractor:
    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: str = None):
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2Model

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Wav2Vec2 ({model_name}) on {self.device} ...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Returns mean-pooled last hidden state: (768,) for wav2vec2-base."""
        torch = self.torch
        with torch.no_grad():
            inputs = self.processor(
                audio, sampling_rate=sr, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0)
        return embedding.cpu().numpy()  # (768,)


# ── Dataset scanner ───────────────────────────────────────────────────────────

def scan_dataset(audio_dir: str) -> pd.DataFrame:
    """
    Scans processed (or raw) audio dir organized as <dir>/<emotion>/<file>.wav.
    Returns a DataFrame with columns: path, emotion, actor, gender.
    """
    records = []
    for wav_file in sorted(Path(audio_dir).rglob("*.wav")):
        emotion = wav_file.parent.name
        if emotion not in LABEL2IDX:
            continue
        parts = wav_file.stem.split("-")
        actor = int(parts[6]) if len(parts) >= 7 else -1
        records.append({
            "path": str(wav_file),
            "emotion": emotion,
            "actor": actor,
            "gender": "male" if actor % 2 == 1 else "female",
        })
    return pd.DataFrame(records)


def make_train_test_split(df: pd.DataFrame, test_size: float = 0.2,
                          random_state: int = 42) -> pd.DataFrame:
    """Stratified split by emotion, stored as a 'split' column."""
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=df["emotion"],
        random_state=random_state
    )
    df = df.copy()
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df


# ── Main extraction pipeline ─────────────────────────────────────────────────

def run_extraction(
    audio_dir: str = None,
    features_dir: str = "data/features",
    config_path: str = "configs/config.yaml",
    extract_wav2vec2: bool = True,
) -> None:
    cfg = load_config(config_path)
    fe_cfg = cfg["feature_extraction"]
    model_cfg = cfg["model"]

    # Use processed dir if it has content, otherwise fall back to raw
    if audio_dir is None:
        processed = Path(cfg["dataset"]["processed_dir"])
        raw = Path(cfg["dataset"]["raw_dir"])
        audio_dir = str(processed) if any(processed.rglob("*.wav")) else str(raw)

    print(f"Scanning audio from: {audio_dir}")
    df = scan_dataset(audio_dir)
    if df.empty:
        raise RuntimeError(
            f"No .wav files found under {audio_dir}. "
            "Run src/download_dataset.py and src/preprocessing.py first."
        )

    df = make_train_test_split(
        df,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_state"],
    )
    print(f"  Total: {len(df)} files | "
          f"Train: {(df.split=='train').sum()} | Test: {(df.split=='test').sum()}")

    out_dir = Path(features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mfcc_cfg = fe_cfg["mfcc"]
    mfcc_features = []
    labels = []

    # ── MFCC extraction ───────────────────────────────────────────────────────
    print("\nExtracting MFCC features ...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio, sr = librosa.load(row["path"], sr=cfg["audio_preprocessing"]["target_sr"])
        feat = extract_mfcc(
            audio, sr,
            n_mfcc=mfcc_cfg["n_mfcc"],
            n_fft=mfcc_cfg["n_fft"],
            hop_length=mfcc_cfg["hop_length"],
        )
        mfcc_features.append(feat)
        labels.append(LABEL2IDX[row["emotion"]])

    mfcc_arr = np.array(mfcc_features, dtype=np.float32)   # (N, 240)
    labels_arr = np.array(labels, dtype=np.int32)           # (N,)

    np.save(out_dir / "mfcc_features.npy", mfcc_arr)
    np.save(out_dir / "labels.npy", labels_arr)
    print(f"  MFCC features: {mfcc_arr.shape}")

    # ── Wav2Vec2 extraction ───────────────────────────────────────────────────
    if extract_wav2vec2:
        extractor = Wav2Vec2Extractor(model_name=fe_cfg["wav2vec2"]["model_name"])
        print("\nExtracting Wav2Vec2 embeddings ...")
        wav2vec2_features = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            audio, sr = librosa.load(
                row["path"], sr=cfg["audio_preprocessing"]["target_sr"]
            )
            emb = extractor.extract(audio, sr=sr)
            wav2vec2_features.append(emb)

        w2v_arr = np.array(wav2vec2_features, dtype=np.float32)  # (N, 768)
        np.save(out_dir / "wav2vec2_features.npy", w2v_arr)
        print(f"  Wav2Vec2 features: {w2v_arr.shape}")

    # ── Metadata & label map ──────────────────────────────────────────────────
    df.reset_index(drop=True).to_csv(out_dir / "metadata.csv", index=False)
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(LABEL2IDX, f, indent=2)

    print(f"\nAll features saved to: {out_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract MFCC and Wav2Vec2 features")
    parser.add_argument("--audio_dir", default=None,
                        help="Override audio source dir (default: auto-detect)")
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--no_wav2vec2", action="store_true",
                        help="Skip Wav2Vec2 (faster, for testing MFCC only)")
    args = parser.parse_args()

    run_extraction(
        audio_dir=args.audio_dir,
        features_dir=args.features_dir,
        extract_wav2vec2=not args.no_wav2vec2,
    )
