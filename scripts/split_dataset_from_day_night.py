# scripts/split_dataset_from_day_night.py
import os
import shutil
import random

SOURCE_DIRS = ["dataset/daytime", "dataset/nighttime"]
DEST_ROOT = "dataset"
CLASSES = ["car", "bus", "truck", "motorcycle", "human"]
TRAIN_RATIO = 0.8
SEED = 42


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_images():
    all_images = {cls: [] for cls in CLASSES}
    for source in SOURCE_DIRS:
        for cls in CLASSES:
            cls_path = os.path.join(source, cls)
            if os.path.exists(cls_path):
                for fname in os.listdir(cls_path):
                    if fname.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
                    ):
                        full_path = os.path.join(cls_path, fname)
                        all_images[cls].append(full_path)
    return all_images


def split_and_copy(all_images):
    for cls, images in all_images.items():
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for subset, subset_images in [("train", train_images), ("val", val_images)]:
            out_dir = os.path.join(DEST_ROOT, subset, cls)
            ensure_dir(out_dir)
            for src_path in subset_images:
                fname = os.path.basename(src_path)
                dst_path = os.path.join(out_dir, fname)
                shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    random.seed(SEED)
    all_images = collect_images()
    split_and_copy(all_images)
    print("✅ Dataset split into train/val and copied successfully.")
