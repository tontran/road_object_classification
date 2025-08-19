# roc/data/label_cli.py

import argparse
import json
import os
from pathlib import Path
from collections import deque

import cv2
import pandas as pd

DEFAULT_CLASSES = ["car", "bus", "truck", "motorcycle", "human"]
STATE_FILE = ".label_state.json"  # saved next to the CSV

KEYMAP = {
    ord("1"): 0,
    ord("2"): 1,
    ord("3"): 2,
    ord("4"): 3,
    ord("5"): 4,
}


def load_state(csv_path: Path):
    state_path = csv_path.parent / STATE_FILE
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(csv_path: Path, state: dict):
    state_path = csv_path.parent / STATE_FILE
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f)


def fit_to_screen(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def cmd_label(args):
    class_names = DEFAULT_CLASSES
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "label" not in df.columns:
        raise SystemExit("CSV must have columns: filename,label")

    def norm_path(p):
        p = os.path.expanduser(str(p))
        return str(Path(p).resolve()) if os.path.exists(p) else p

    df["filename"] = df["filename"].map(norm_path)
    unlabeled = df.index[
        df["label"].isna() | (df["label"].astype(str).str.strip() == "")
    ].tolist()
    if not unlabeled:
        print("🎉 Nothing to label.")
        return

    state = load_state(csv_path)
    pos = state.get("start_pos", 0)
    pos = max(0, min(pos, len(unlabeled) - 1))

    undo = deque()
    labeled_since_save = 0
    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)

    while 0 <= pos < len(unlabeled):
        idx = unlabeled[pos]
        img_path = df.at[idx, "filename"]
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Cannot read: {img_path}")
            pos += 1
            continue

        view = fit_to_screen(img.copy())
        bar = 40
        overlay = view.copy()
        cv2.rectangle(overlay, (0, 0), (view.shape[1], bar), (0, 0, 0), -1)
        view = cv2.addWeighted(overlay, 0.6, view, 0.4, 0)
        help_text = "1:car | 2:bus | 3:truck | 4:motorcycle | 5:human | s:skip | u:undo | q:quit"
        cv2.putText(
            view,
            help_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            os.path.basename(img_path),
            (10, view.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("label", view)
        k = cv2.waitKey(0) & 0xFF

        if k == ord("q"):
            break
        if k == ord("s"):
            pos += 1
            continue
        if k == ord("u"):
            if undo:
                i, prev = undo.pop()
                df.at[i, "label"] = prev
                df.to_csv(csv_path, index=False)
                save_state(csv_path, {"start_pos": pos})
                print("↩️  Undo saved.")
            else:
                print("ℹ️ Nothing to undo.")
            continue
        if k in KEYMAP:
            cls_idx = KEYMAP[k]
            if 0 <= cls_idx < len(class_names):
                prev = df.at[idx, "label"] if "label" in df.columns else ""
                df.at[idx, "label"] = class_names[cls_idx]
                undo.append((idx, prev))
                labeled_since_save += 1
                print(f"✔️  {img_path} → {class_names[cls_idx]}")
                if labeled_since_save >= max(1, getattr(args, "autosave_every", 1)):
                    df.to_csv(csv_path, index=False)
                    save_state(csv_path, {"start_pos": pos})
                    labeled_since_save = 0
                pos += 1
                continue
        print("ℹ️ Use 1..5, s, u, q.")

    df.to_csv(csv_path, index=False)
    save_state(csv_path, {"start_pos": pos})
    cv2.destroyAllWindows()
    print("✅ Saved CSV. Done.")


def main():
    ap = argparse.ArgumentParser(
        description="Quick labeling CLI for frames using number keys."
    )
    ap.add_argument(
        "--csv", required=True, help="Path to labels.csv (columns: filename,label)"
    )
    ap.add_argument(
        "--classes",
        nargs="*",
        default=DEFAULT_CLASSES,
        help="Class names in order of keys 1..N",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=None,
        help="Optional start row index (0-based) among UNLABELED rows",
    )
    ap.add_argument(
        "--autosave-every",
        type=int,
        default=1,
        help="Autosave frequency (every N labels)",
    )
    args = ap.parse_args()

    class_names = args.classes
    if len(class_names) < 2:
        raise SystemExit("Need at least 2 classes.")
    print(f"Classes (1..{len(class_names)}): {class_names}")

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "label" not in df.columns:
        raise SystemExit("CSV must have columns: filename,label")

    # Normalize paths (expand ~ and make absolute if possible)
    def norm_path(p):
        p = str(p)
        p = os.path.expanduser(p)
        return str(Path(p).resolve()) if os.path.exists(os.path.expanduser(p)) else p

    df["filename"] = df["filename"].map(norm_path)

    # Work only on rows that are unlabeled (label empty or NaN)
    unlabeled_mask = df["label"].isna() | (df["label"].astype(str).str.strip() == "")
    unlabeled_idx = df[unlabeled_mask].index.tolist()

    if not unlabeled_idx:
        print("🎉 Nothing to label. All rows already labeled.")
        return

    # Determine starting position
    state = load_state(csv_path)
    start_pos = args.start if args.start is not None else state.get("start_pos", 0)
    start_pos = max(0, min(start_pos, len(unlabeled_idx) - 1))
    print(f"Starting at unlabeled item #{start_pos+1} / {len(unlabeled_idx)}")

    undo_stack = deque()  # holds tuples (df_index, previous_label)
    labeled_since_save = 0

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)

    pos = start_pos
    while 0 <= pos < len(unlabeled_idx):
        row_idx = unlabeled_idx[pos]
        row = df.loc[row_idx]
        img_path = row["filename"]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read image: {img_path} (skipping)")
            pos += 1
            continue

        # Render overlay with instructions
        view = fit_to_screen(img.copy())
        overlay = view.copy()
        bar = 40
        cv2.rectangle(overlay, (0, 0), (view.shape[1], bar), (0, 0, 0), -1)
        alpha = 0.6
        view = cv2.addWeighted(overlay, alpha, view, 1 - alpha, 0)

        key_help = " | ".join(
            [f"{i+1}:{name}" for i, name in enumerate(class_names[:5])]
        )
        help_text = f"{key_help}   s:skip   u:undo   q:quit"
        cv2.putText(
            view,
            help_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Show path at bottom
        cv2.putText(
            view,
            os.path.basename(img_path),
            (10, view.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("label", view)
        k = cv2.waitKey(0) & 0xFF

        if k == ord("q"):
            print("👋 Quit requested.")
            break
        elif k == ord("s"):
            # skip this one (stay unlabeled)
            pos += 1
            continue
        elif k == ord("u"):
            # Undo last label in this session
            if undo_stack:
                idx, prev = undo_stack.pop()
                df.at[idx, "label"] = prev
                # Move cursor back to this image to relabel if you want
                try:
                    pos = unlabeled_idx.index(idx)
                except ValueError:
                    # If it wasn't unlabeled originally, just continue
                    pass
                # Save immediately after undo
                df.to_csv(csv_path, index=False)
                print("↩️  Undo applied and saved.")
            else:
                print("ℹ️ Nothing to undo.")
            continue
        elif k in KEYMAP and KEYMAP[k] < len(class_names):
            chosen = class_names[KEYMAP[k]]
            prev_label = df.at[row_idx, "label"] if "label" in df.columns else ""
            df.at[row_idx, "label"] = chosen
            undo_stack.append((row_idx, prev_label))
            labeled_since_save += 1
            print(f"✔️  Labeled as: {chosen}")
        else:
            print("ℹ️ Invalid key. Use 1-5, s, u, q.")
            continue

        # Autosave
        if labeled_since_save >= max(1, args.autosave_every):
            df.to_csv(csv_path, index=False)
            # Persist cursor (position in the unlabeled list)
            save_state(csv_path, {"start_pos": pos})
            labeled_since_save = 0

        # Next unlabeled
        pos += 1

    # Final save
    df.to_csv(csv_path, index=False)
    save_state(csv_path, {"start_pos": pos})
    cv2.destroyAllWindows()
    print("✅ Saved CSV. Done.")


if __name__ == "__main__":
    main()
