import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from dataset import get_loaders
from model import SimpleCNN, build_resnet18

CONFIGS = {
    "simple_cnn": dict(build=SimpleCNN, epochs=8, lr=1e-3),
    "resnet18":   dict(build=build_resnet18, epochs=5, lr=3e-4),
}
CKPT = Path("checkpoints"); CKPT.mkdir(exist_ok=True)

def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += labels.size(0)
    return total_loss / n, correct / n

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "simple_cnn"
    cfg = CONFIGS[name]
    device = "cuda"
    train_loader, val_loader, _ = get_loaders()
    model = cfg["build"]().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda")

    print(f"training {name} for {cfg['epochs']} epochs")
    best_acc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        print(f"epoch {epoch}/{cfg['epochs']} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - start:.0f}s")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), CKPT / f"{name}.pt")
            print("  saved new best model")
    print(f"best val acc: {best_acc:.3f}")

if __name__ == "__main__":
    main()