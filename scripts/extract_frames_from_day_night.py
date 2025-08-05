import os
import cv2

# Set your raw video source folder
VIDEO_ROOT = (
    r"C:\Users\ton.tran\OneDrive - Visual Defence Inc\Projects\raw_data\road_video"
)
OUTPUT_ROOT = "raw_frames"  # inside your project folder
FRAME_INTERVAL = 30  # Save every 30th frame (~1/sec at 30 FPS)


def extract_frames_from_video(video_path, output_dir, interval):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    os.makedirs(output_dir, exist_ok=True)

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
    print(f"✅ Extracted {saved_count} frames → {output_dir}")


def process_day_night_videos():
    # for lighting in ["daytime", "nighttime"]:
    for lighting in ["nighttime"]: # skipping some daytimes footage and jump to night time
        folder_path = os.path.join(VIDEO_ROOT, lighting)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(folder_path, filename)
                video_name = os.path.splitext(filename)[0]
                output_dir = os.path.join(OUTPUT_ROOT, lighting, video_name)
                extract_frames_from_video(video_path, output_dir, FRAME_INTERVAL)


if __name__ == "__main__":
    process_day_night_videos()
