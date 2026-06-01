"""
Audio preprocessing: resampling, normalization, silence removal.
All parameters are read from configs/config.yaml.
"""

import numpy as np
import librosa
import soundfile as sf
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


def remove_silence(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.01,
    frame_length: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    intervals = librosa.effects.split(
        audio,
        top_db=int(-20 * np.log10(threshold + 1e-9)),
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if len(intervals) == 0:
        return audio
    return np.concatenate([audio[s:e] for s, e in intervals])


def pad_or_trim(audio: np.ndarray, sr: int, max_duration: float = 5.0) -> np.ndarray:
    max_samples = int(sr * max_duration)
    if len(audio) > max_samples:
        return audio[:max_samples]
    if len(audio) < max_samples:
        return np.pad(audio, (0, max_samples - len(audio)))
    return audio


def preprocess_file(
    input_path: str,
    output_path: str,
    target_sr: int = 16000,
    max_duration: float = 5.0,
    silence_threshold: float = 0.01,
    normalize: bool = True,
) -> None:
    audio, sr = load_audio(input_path, target_sr)
    audio = remove_silence(audio, sr, threshold=silence_threshold)
    if normalize:
        audio = normalize_audio(audio)
    audio = pad_or_trim(audio, sr, max_duration)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)


def preprocess_dataset(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    config_path: str = "configs/config.yaml",
) -> None:
    cfg = load_config(config_path)
    pp = cfg["audio_preprocessing"]

    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    wav_files = list(raw_path.rglob("*.wav"))

    print(f"Preprocessing {len(wav_files)} files ...")
    for i, wav_file in enumerate(wav_files, 1):
        rel = wav_file.relative_to(raw_path)
        out_file = proc_path / rel
        preprocess_file(
            str(wav_file),
            str(out_file),
            target_sr=pp["target_sr"],
            max_duration=pp["max_duration"],
            silence_threshold=pp["silence_threshold"],
            normalize=pp["normalize"],
        )
        if i % 100 == 0 or i == len(wav_files):
            print(f"  {i}/{len(wav_files)}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    preprocess_dataset()
