import torch
import torch.nn as nn
from torchvision import models

def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

def build_resnet18(num_classes=38):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # pretrained on ImageNet
    model.fc = nn.Linear(model.fc.in_features, num_classes)                 # new last layer: 512 -> 38
    return model

if __name__ == "__main__":
    for name, model in [("SimpleCNN", SimpleCNN()), ("ResNet18", build_resnet18())]:
        out = model(torch.randn(2, 3, 224, 224))
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:10s} output {tuple(out.shape)}  parameters {params:,}")