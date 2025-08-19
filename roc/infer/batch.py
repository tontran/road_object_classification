# roc/infer/batch.py
"""
desktop sanity check
"""
import os, glob, torch
from PIL import Image
from torchvision import transforms


def _norm():
    return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def cmd_infer(args):
    model = torch.jit.load(args.model, map_location="cpu").eval()
    tfm = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), _norm()]
    )

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.images, e))
    print(f"Found {len(files)} images.")

    for f in files[:20]:  # limit to 20 for quick check
        img = Image.open(f).convert("RGB")
        x = tfm(img).unsqueeze(0)
        with torch.no_grad():
            y = model(x)[0]
        i = int(torch.argmax(y).item())
        conf = float(torch.softmax(y, 0)[i])
        print(f"{os.path.basename(f)} → class_id={i} conf={conf:.3f}")
