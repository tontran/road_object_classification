# scripts/copy_split_from_onedrive_csv.py

import os, sys, csv, shutil, random
from pathlib import Path

RAW_FRAMES_ROOT = Path("/Users/tontran/OneDrive/OneDrive - Visual Defence Inc/Projects/road_object_shared/raw_frames")
LABELS_CSV      = Path("labels.csv")
DEST_ROOT       = Path("dataset")
TRAIN_RATIO     = 0.8
SEED            = 42
USE_HARDLINKS   = True  # set False if cross-filesystem errors

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def copy_or_link(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if USE_HARDLINKS:
        try:
            if not dst.exists():
                os.link(src, dst)
            return
        except OSError:
            pass
    if not dst.exists():
        shutil.copy2(src, dst)

def main():
    # Read labeled rows only
    by_class = {}
    with LABELS_CSV.open(newline="", encoding="utf-8") as fp:
        r = csv.DictReader(fp)
        for row in r:
            label = (row.get("label") or "").strip()
            fname = (row.get("filename") or "").strip()
            if not label or not fname:
                continue
            p = Path(fname)
            if not p.is_absolute():
                matches = list(RAW_FRAMES_ROOT.rglob(p.name))
                if not matches:
                    print(f"⚠️ Not found under RAW_FRAMES_ROOT: {fname}")
                    continue
                p = matches[0]
            if not p.exists():
                print(f"⚠️ File missing: {p}")
                continue
            by_class.setdefault(label, []).append(p)

    random.seed(SEED)
    train_root = DEST_ROOT / "train"
    val_root   = DEST_ROOT / "val"

    print("📦 Splitting and materializing dataset...")
    for cls, files in by_class.items():
        random.shuffle(files)
        n = len(files)
        k = int(n * TRAIN_RATIO)
        train_files, val_files = files[:k], files[k:]

        for src in train_files:
            dst = train_root / cls / src.name
            copy_or_link(src, dst)
        for src in val_files:
            dst = val_root / cls / src.name
            copy_or_link(src, dst)

        print(f"  {cls:<12} train={len(train_files):>5} | val={len(val_files):>5}")

    print("✅ Done. You can now run:  python train.py")

if __name__ == "__main__":
    main()

    