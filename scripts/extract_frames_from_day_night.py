# scripts/extract_frames_from_day_night.py
"""
Usage:
# Nighttime only
python scripts/extract_frames_from_day_night.py --lighting nighttime

# Daytime only
python scripts/extract_frames_from_day_night.py --lighting daytime

# All videos
python scripts/extract_frames_from_day_night.py --lighting all

"""
import os
import argparse
from multiprocessing import Pool, cpu_count
import cv2
from helpers.file_utils import get_existing_frames

VIDEO_ROOT = (
    r"C:\Users\ton.tran\OneDrive - Visual Defence Inc\Projects\raw_data\road_video"
)
OUTPUT_ROOT = "raw_frames"
FRAME_INTERVAL = 30
EXPECTED_FRAMES = 48


def extract_frames_from_video(video_path, output_dir, interval):
    os.makedirs(output_dir, exist_ok=True)
    existing_frames = get_existing_frames(output_dir)
    saved_count = len(existing_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_count * interval)
    frame_count = saved_count * interval

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % interval == 0:
            filename = f"frame_{saved_count:05}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Resumed and extracted up to frame {saved_count} → {output_dir}")


def should_skip(output_dir):
    existing_frames = get_existing_frames(output_dir)
    return len(existing_frames) >= EXPECTED_FRAMES


def process_single_video(args):
    video_path, output_dir = args
    video_name = os.path.basename(video_path).split(".")[0]

    if should_skip(output_dir):
        print(f"⚠️  Skipping {video_name}: already has {EXPECTED_FRAMES} frames.")
        return

    existing_count = len(get_existing_frames(output_dir))
    print(f"🟡 Found {existing_count} existing frames in {output_dir}")
    extract_frames_from_video(video_path, output_dir, FRAME_INTERVAL)


def process_lighting_type(lighting_type):
    folder_path = os.path.join(VIDEO_ROOT, lighting_type)
    tasks = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        video_path = os.path.join(folder_path, filename)
        video_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(OUTPUT_ROOT, lighting_type, video_name)
        tasks.append((video_path, output_dir))

    print(f"⚙️ Starting multiprocessing for {lighting_type} ({len(tasks)} videos)...")
    with Pool(processes=cpu_count() - 1 or 1) as pool:
        pool.map(process_single_video, tasks)


def main():
    parser = argparse.ArgumentParser(description="Extract frames from road video")
    parser.add_argument(
        "--lighting",
        type=str,
        choices=["daytime", "nighttime", "all"],
        default="all",
        help="Lighting condition to process (daytime, nighttime, all)",
    )
    args = parser.parse_args()

    if args.lighting == "all":
        for lighting in ["daytime", "nighttime"]:
            process_lighting_type(lighting)
    else:
        process_lighting_type(args.lighting)


if __name__ == "__main__":
    main()
