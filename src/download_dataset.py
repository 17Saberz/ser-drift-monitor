"""
Download and extract the RAVDESS dataset from Zenodo.
RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
https://zenodo.org/record/1188976

File naming convention:
  03-01-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav
  Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry,
           06=fearful, 07=disgust, 08=surprised
  Intensity: 01=normal, 02=strong
  Actor: 01-24 (odd=male, even=female)
"""

import os
import zipfile
import urllib.request
from pathlib import Path

RAVDESS_URLS = {
    "speech": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
    "song":   "https://zenodo.org/record/1188976/files/Audio_Song_Actors_01-24.zip",
}

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url.split('/')[-1]} ...")

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size
        print(f"\r  {min(pct, 100):.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def flatten_to_emotion_dirs(raw_dir: Path) -> None:
    """
    RAVDESS extracts into Actor_XX subdirs.
    Flatten into data/raw/<emotion>/<file>.wav for easier loading.
    """
    for wav_file in raw_dir.rglob("*.wav"):
        parts = wav_file.stem.split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        emotion_label = EMOTION_MAP.get(emotion_code, "unknown")
        target_dir = raw_dir / emotion_label
        target_dir.mkdir(exist_ok=True)
        dest = target_dir / wav_file.name
        if not dest.exists():
            wav_file.rename(dest)

    # Remove now-empty Actor_XX dirs
    for actor_dir in raw_dir.glob("Actor_*"):
        if actor_dir.is_dir():
            try:
                actor_dir.rmdir()
            except OSError:
                pass


def main(subset: str = "speech", raw_dir: str = "data/raw") -> None:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    if subset not in RAVDESS_URLS:
        raise ValueError(f"subset must be one of {list(RAVDESS_URLS)}")

    url = RAVDESS_URLS[subset]
    zip_path = raw_path / url.split("/")[-1]

    if not zip_path.exists():
        download_file(url, zip_path)
    else:
        print(f"Zip already exists: {zip_path}")

    extract_zip(zip_path, raw_path)
    flatten_to_emotion_dirs(raw_path)

    # Remove zip after extraction
    zip_path.unlink(missing_ok=True)

    counts = {d.name: len(list(d.glob("*.wav"))) for d in raw_path.iterdir() if d.is_dir()}
    print("\nDownload complete. File counts per emotion:")
    for emotion, count in sorted(counts.items()):
        print(f"  {emotion:<12} {count} files")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download RAVDESS dataset")
    parser.add_argument("--subset", choices=["speech", "song"], default="speech")
    parser.add_argument("--raw_dir", default="data/raw")
    args = parser.parse_args()

    main(subset=args.subset, raw_dir=args.raw_dir)
