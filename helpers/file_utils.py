# helpers/file_utils.py
import os
import glob


def get_existing_frames(output_dir: str):
    if not os.path.exists(output_dir):
        return []
    return sorted(glob.glob(os.path.join(output_dir, "frame_*.jpg")))
