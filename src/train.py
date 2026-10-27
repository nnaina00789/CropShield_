import time
from pathlib import Path
import torch
import torch.nn as nn
from dataset import get_loaders
from model import SimpleCNN

EPOCHS = 8
LR = 1e-3
CKPT = Path("checkpoints"); CKPT.mkdir(exist_ok=True)

def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)                      # train mode or eval mode
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):     # no gradients needed for validation
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)               # 1. forward pass
                loss = loss_fn(logits, labels)       # 2. loss
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()        # 3. backward pass
                scaler.step(optimizer)               # 4. optimizer step
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += labels.size(0)
    return total_loss / n, correct / n

def main():
    device = "cuda"
    train_loader, val_loader, _ = get_loaders()
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn, device)
        print(f"epoch {epoch}/{EPOCHS} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - start:.0f}s")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), CKPT / "simple_cnn.pt")
            print("  saved new best model")
    print(f"best val acc: {best_acc:.3f}")

if __name__ == "__main__":
    main()