# roc/data/split.py
"""
CSV → dataset/
"""
import os, csv, random, shutil
from pathlib import Path
from roc.io.file_utils import ensure_dir


def _copy_or_link(src: Path, dst: Path, hardlinks: bool):
    ensure_dir(str(dst.parent))
    if hardlinks:
        try:
            if not dst.exists():
                os.link(src, dst)
            return
        except OSError:
            pass
    if not dst.exists():
        shutil.copy2(src, dst)


def cmd_split(args):
    labels = Path(args.labels).expanduser()
    root = Path(args.raw_frames).expanduser()
    dest = Path(args.dest)

    by_class = {}
    with labels.open(newline="", encoding="utf-8") as fp:
        r = csv.DictReader(fp)
        for row in r:
            label = (row.get("label") or "").strip()
            fname = (row.get("filename") or "").strip()
            if not label or not fname:
                continue
            p = Path(fname)
            if not p.is_absolute():
                matches = list(root.rglob(p))
                if not matches:
                    print(f"⚠️ not found under {root}: {fname}")
                    continue
                p = matches[0]
            if not p.exists():
                print(f"⚠️ missing: {p}")
                continue
            by_class.setdefault(label, []).append(p)

    random.seed(42)
    for cls, files in by_class.items():
        random.shuffle(files)
        k = int(len(files) * args.train_ratio)
        train_files, val_files = files[:k], files[k:]

        for s in train_files:
            _copy_or_link(s, dest / "train" / cls / s.name, args.hardlinks)
        for s in val_files:
            _copy_or_link(s, dest / "val" / cls / s.name, args.hardlinks)

        print(f"{cls:<12} train={len(train_files):>5} | val={len(val_files):>5}")
    print("✅ Split complete.")
