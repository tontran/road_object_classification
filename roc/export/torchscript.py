# roc/export/torchscript.py
import os, torch
from torchvision import models
from torch import nn


def cmd_export(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, args.num_classes)
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    ex = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, ex)
    traced.save(args.out)
    print(f"✅ TorchScript saved → {args.out}")
