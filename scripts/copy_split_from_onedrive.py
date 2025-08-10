# scripts/copy_split_from_onedrive.py

import os, sys, shutil, random, argparse, glob
from pathlib import Path

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_or_link(src: Path, dst: Path, use_hardlinks: bool):
    ensure_dir(dst.parent)
    if use_hardlinks:
        try:
            if dst.exists():
                return
            os.link(src, dst)
            return
        except OSError:
            # Fallback to copy if hardlink fails (different filesystem, cloud FS, etc.)
            pass
    if not dst.exists():
        shutil.copy2(src, dst)

def collect_from_classes(day_root: Path, night_root: Path, classes: list[str]):
    """
    Merge daytime + nighttime folder structures that look like:
      <root>/
        daytime/<class>/*.jpg
        nighttime/<class>/*.jpg
    Returns dict[class] -> [Path,...]
    """
    by_class = {c: [] for c in classes}

    for root in [day_root, night_root]:
        if not root.exists():
            continue
        for c in classes:
            cdir = root / c
            if not cdir.exists():
                continue
            # include common image extensions (lower/upper)
            exts = ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp",
                    "*.JPG","*.JPEG","*.PNG","*.BMP","*.TIF","*.TIFF","*.WEBP"]
            files = []
            for e in exts:
                files.extend(cdir.glob(e))
            by_class[c].extend(files)
    return by_class

def collect_from_csv(raw_frames_root: Path, labels_csv: Path):
    """
    labels.csv must have headers: filename,label
    filename can be either a full path or a basename present somewhere under raw_frames_root.
    Returns dict[label] -> [Path,...]
    """
    import csv
    by_class = {}
    for row in csv.DictReader(open(labels_csv, newline='', encoding='utf-8')):
        fname = row.get("filename") or row.get("file") or row.get("path")
        label = row.get("label") or row.get("class") or row.get("category")
        if not fname or not label:
            print(f"⚠️  Skipping row (missing filename/label): {row}")
            continue

        p = Path(fname)
        if not p.is_absolute():
            # search under raw_frames_root for a matching file (slow but robust)
            matches = list(raw_frames_root.rglob(p.name))
            if not matches:
                print(f"⚠️  Not found under raw_frames_root: {fname}")
                continue
            p = matches[0]
        if not p.exists():
            print(f"⚠️  File not found: {p}")
            continue
        by_class.setdefault(label, []).append(p)
    return by_class

def split_and_write(by_class: dict[str, list[Path]],
                    dest_root: Path,
                    train_ratio: float,
                    seed: int,
                    use_hardlinks: bool):
    random.seed(seed)
    train_root = dest_root / "train"
    val_root   = dest_root / "val"

    total_counts = {}
    for c, files in by_class.items():
        files = [f for f in files if f.exists()]
        random.shuffle(files)
        n = len(files)
        split_idx = int(n * train_ratio)

        train_files = files[:split_idx]
        val_files   = files[split_idx:]

        total_counts[c] = {"train": len(train_files), "val": len(val_files)}

        # write
        for src in train_files:
            dst = train_root / c / src.name
            copy_or_link(src, dst, use_hardlinks)

        for src in val_files:
            dst = val_root / c / src.name
            copy_or_link(src, dst, use_hardlinks)

    # simple summary
    print("✅ Split summary:")
    for c, d in total_counts.items():
        print(f"  {c:<12} train={d['train']:>5} | val={d['val']:>5}")

def main():
    parser = argparse.ArgumentParser(description="Copy/split images from OneDrive into dataset/train & dataset/val.")
    parser.add_argument("--mode", choices=["classes", "csv"], required=True,
                        help="classes: expects daytime/nighttime/<class>/... ; csv: labels.csv mapping")
    parser.add_argument("--day-root", type=str,
                        help="Path to daytime class root (required for mode=classes)")
    parser.add_argument("--night-root", type=str,
                        help="Path to nighttime class root (required for mode=classes)")
    parser.add_argument("--raw-frames-root", type=str,
                        help="Path to raw_frames root (required for mode=csv)")
    parser.add_argument("--labels-csv", type=str,
                        help="CSV with columns filename,label (required for mode=csv)")
    parser.add_argument("--dest", type=str, default="dataset",
                        help="Destination root where train/ and val/ will be written (default: dataset in repo)")
    parser.add_argument("--classes", type=str, nargs="*", default=["car","bus","truck","motorcycle","human"],
                        help="Class list (only for mode=classes, default 5 classes)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train split ratio (default 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-hardlinks", action="store_true",
                        help="Use hardlinks instead of copying (faster, no duplication; may fail across filesystems)")
    args = parser.parse_args()

    dest_root = Path(args.dest)
    ensure_dir(dest_root)

    if args.mode == "classes":
        if not args.day_root or not args.night_root:
            print("❌ mode=classes requires --day-root and --night-root")
            sys.exit(1)
        day = Path(args.day_root).expanduser()
        night = Path(args.night_root).expanduser()
        by_class = collect_from_classes(day, night, args.classes)

    else:  # mode == "csv"
        if not args.raw_frames_root or not args.labels_csv:
            print("❌ mode=csv requires --raw-frames-root and --labels-csv")
            sys.exit(1)
        raw_root = Path(args.raw_frames_root).expanduser()
        labels_csv = Path(args.labels_csv).expanduser()
        by_class = collect_from_csv(raw_root, labels_csv)

    split_and_write(by_class, dest_root, args.train_ratio, args.seed, args.use_hardlinks)

if __name__ == "__main__":
    main()