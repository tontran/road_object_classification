# roc/train/train.py
"""
callable from CLI; reuses your logic
"""
import os, torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def _norm():
    return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def cmd_train(args):
    data_root = args.data
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    os.makedirs("models", exist_ok=True)

    tfm = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), _norm()]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(train_ds.classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_loss = float("inf")
    patience = 5
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        tloss, tcorrect, ttotal = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            o = model(x)
            loss = criterion(o, y)
            loss.backward()
            optimizer.step()
            tloss += loss.item() * x.size(0)
            tcorrect += (o.argmax(1) == y).sum().item()
            ttotal += y.size(0)
        train_loss = tloss / ttotal
        train_acc = tcorrect / ttotal

        # val
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                o = model(x)
                loss = criterion(o, y)
                vloss += loss.item() * x.size(0)
                vcorrect += (o.argmax(1) == y).sum().item()
                vtotal += y.size(0)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(o.argmax(1).cpu().numpy().tolist())
        val_loss = vloss / vtotal
        val_acc = vcorrect / vtotal

        scheduler.step(val_loss)
        print(
            f"[{epoch+1}/{args.epochs}] train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_loss < best_loss:
            torch.save(model.state_dict(), args.out)
            best_loss = val_loss
            no_improve = 0
            print(f"✅ Saved best → {args.out}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("🛑 Early stopping.")
                break

    # final metrics
    print("\nClassification Report:")
    print(
        classification_report(y_true, y_pred, target_names=val_loader.dataset.classes)
    )
