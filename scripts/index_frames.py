# scripts/index_frames.py

import os, csv
from pathlib import Path

# Update these to your OneDrive locations on Mac
DAY_DIR  = Path("/Users/tontran/OneDrive/OneDrive - Visual Defence Inc/Projects/road_object_shared/raw_frames/daytime")
NIGHT_DIR= Path("/Users/tontran/OneDrive/OneDrive - Visual Defence Inc/Projects/road_object_shared/raw_frames/nighttime")
OUT_CSV  = Path("labels.csv")  # in repo root

def list_frames(root: Path):
    # accepts frames under root/<video_name>/frame_*.jpg
    if not root.exists(): return []
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".JPG",".PNG",".JPEG",".BMP",".TIF",".TIFF",".WEBP"}
    frames = []
    for vid_dir in sorted(root.iterdir()):
        if not vid_dir.is_dir(): continue
        for f in sorted(vid_dir.iterdir()):
            if f.suffix in exts:
                frames.append(f)
    return frames

def main():
    rows = []
    for f in list_frames(DAY_DIR) + list_frames(NIGHT_DIR):
        # write absolute path to be robust
        rows.append({"filename": str(f.resolve()), "label": ""})  # label to fill later

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["filename","label"])
        w.writeheader()
        w.writerows(rows)

    print(f"✅ Indexed {len(rows)} frames into {OUT_CSV}")

if __name__ == "__main__":
    main()
