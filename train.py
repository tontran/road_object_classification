# train.py
import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from datetime import datetime

# ==== CONFIG ====
train_dir = "dataset/train"
val_dir = "dataset/val"
checkpoint_path = "checkpoints/mobilenetv3_checkpoint.pth"
batch_size = 32
num_epochs = 20
num_classes = 5
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# ==== TRANSFORMS ====
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# ==== DATASETS ====
train_ds = datasets.ImageFolder(train_dir, transform=transform)
val_ds = datasets.ImageFolder(val_dir, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# ==== MODEL ====
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(device)

# ==== LOSS & OPTIMIZER ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==== RESUME FROM CHECKPOINT ====
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"🔁 Resuming training from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"✅ Resumed from epoch {start_epoch}")

# ==== TRAIN LOOP ====
try:
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # ==== VALIDATION ====
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {running_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {acc:.4f}"
        )

        # ==== SAVE CHECKPOINT ====
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"💾 Checkpoint saved at epoch {epoch}")

except KeyboardInterrupt:
    print("\n🛑 Training interrupted by user. Saving checkpoint...")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"💾 Last checkpoint saved at epoch {epoch}")
