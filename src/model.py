import torch
import torch.nn as nn

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

if __name__ == "__main__":
    model = SimpleCNN()
    x = torch.randn(32, 3, 224, 224)  # fake batch, same shape as our real ones
    print("input:        ", tuple(x.shape))
    for i, blk in enumerate(model.features, 1):
        x = blk(x)
        print(f"after block {i}:", tuple(x.shape))
    x = model.pool(x).flatten(1)
    print("after pooling:", tuple(x.shape))
    print("output:       ", tuple(model.classifier(x).shape))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")