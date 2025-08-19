# roc/io/file_utils.py
import os, glob


def list_frames(output_dir: str):
    if not os.path.exists(output_dir):
        return []
    pattern = os.path.join(output_dir, "frame_*.jpg")
    return sorted(glob.glob(pattern))


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
