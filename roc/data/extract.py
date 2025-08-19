# roc/data/extract.py
"""
resume + skip
"""
import os
import cv2
from roc.io.file_utils import list_frames, ensure_dir

EXPECTED_FRAMES = 48  # guard; still resume-safe beyond this


def extract_one(video_path, output_dir, interval: int):
    ensure_dir(output_dir)
    existing = list_frames(output_dir)
    saved_count = len(existing)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_count * interval)
    frame_idx = saved_count * interval

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            out = os.path.join(output_dir, f"frame_{saved_count:05}.jpg")
            if not os.path.exists(out):
                cv2.imwrite(out, frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"✅ {os.path.basename(output_dir)} → now has {saved_count} frames")


def cmd_extract(args):
    video_root = args.video_root
    out_root = args.output_root
    lights = ["daytime", "nighttime"] if args.lighting == "all" else [args.lighting]

    for lighting in lights:
        vr = os.path.join(video_root, lighting)
        if not os.path.isdir(vr):
            print(f"ℹ️ Skip missing: {vr}")
            continue
        for fn in sorted(os.listdir(vr)):
            if not fn.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue
            video_path = os.path.join(vr, fn)
            vid = os.path.splitext(fn)[0]
            out_dir = os.path.join(out_root, lighting, vid)

            existing = list_frames(out_dir)
            if len(existing) >= EXPECTED_FRAMES:
                print(f"⚠️  Skip {vid}: already has {len(existing)} frames")
                continue

            extract_one(video_path, out_dir, args.interval)
