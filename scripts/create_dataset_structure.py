import os

def create_folder_structure():
    base_dir = "dataset"
    subfolders = ["daytime", "nighttime"]
    class_names = ["car", "bus", "truck", "motorcycle", "human"]

    for sub in subfolders:
        for cls in class_names:
            path = os.path.join(base_dir, sub, cls)
            os.makedirs(path, exist_ok=True)
            print(f"✅ Created: {path}")


if __name__ == "__main__":
    create_folder_structure()
