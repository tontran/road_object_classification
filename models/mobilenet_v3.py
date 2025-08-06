# models/mobilenet_v3.py
import torchvision.models as models
import torch.nn as nn


def get_mobilenet_v3_small(num_classes: int = 5):
    model = models.mobilenet_v3_small(pretrained=True)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
