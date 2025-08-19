# roc/data/index_frames.py
"""
build labels.csv
"""
import os, csv
from pathlib import Path


def _list_frames(root: Path):
    if not root or not root.exists():
        return []
    exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".JPG",
        ".JPEG",
        ".PNG",
        ".BMP",
        ".TIF",
        ".TIFF",
        ".WEBP",
    }
    frames = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix in exts:
                    frames.append(str(f.resolve()))
    return frames


def cmd_index(args):
    outputs = []
    if args.day:
        outputs += _list_frames(Path(args.day).expanduser())
    if args.night:
        outputs += _list_frames(Path(args.night).expanduser())

    with open(args.out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["filename", "label"])
        w.writeheader()
        for f in outputs:
            w.writerow({"filename": f, "label": ""})
    print(f"✅ Indexed {len(outputs)} frames → {args.out}")
