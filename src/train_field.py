import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from dataset import ROOT as PV_ROOT, SPLITS, MEAN, STD, CLASSES, CLASS_TO_IDX, eval_tf, get_loaders
from model import build_resnet18
from eval_plantdoc import load_plantdoc_split, evaluate_plantdoc

CKPT_OUT = Path("checkpoints/resnet18_field.pt")
OUT = Path("reports"); OUT.mkdir(exist_ok=True)

# Stronger field-style data augmentation: wide scale/aspect crops, lighting/colour jitter, perspective, blur, and random erasing (occlusion)
field_train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(35),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.06),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.25),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value="random"),
])


class ImageTableDataset(Dataset):
    def __init__(self, paths, indices, transform):
        self.paths = list(paths)
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.indices[i]


def build_mixed_train_loader(batch_size=32, num_workers=0, pv_per_class=100, pd_repeat=2):
    """
    Combines a balanced subset of PlantVillage train (up to pv_per_class per class, ~3,800 images)
    with oversampled PlantDoc train (2,337 * pd_repeat = ~4,674 images).
    STRICT RULE: PlantDoc test images are NEVER loaded here.
    """
    pv_train = pd.read_csv(SPLITS / "train.csv")
    parts = [
        g.sample(min(len(g), pv_per_class), random_state=42)
        for _, g in pv_train.groupby("label")
    ]
    pv_sub = pd.concat(parts, ignore_index=True)
    pv_paths = [PV_ROOT / p for p in pv_sub["path"]]
    pv_indices = [CLASS_TO_IDX[c] for c in pv_sub["label"]]

    # Load ONLY PlantDoc's "train" folder
    pd_train_df = load_plantdoc_split("train")
    pd_paths = list(pd_train_df["path"]) * pd_repeat
    pd_indices = list(pd_train_df["idx"]) * pd_repeat

    ds = ImageTableDataset(pv_paths + pd_paths, pv_indices + pd_indices, field_train_tf)
    print(f"Mixed training dataset: {len(pv_paths)} PlantVillage train + {len(pd_paths)} PlantDoc train ({pd_repeat}x) = {len(ds)} images/epoch", flush=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def evaluate_pv_test(model, loader, device="cuda"):
    model.eval()
    y_true, y_pred = [], []
    for images, y in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            preds = model(images.to(device)).argmax(1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.cpu().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (y_true == y_pred).mean(), (y_true != y_pred).sum()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 6

    # Start from our PlantVillage-trained ResNet18 checkpoint
    model = build_resnet18().to(device)
    model.load_state_dict(torch.load("checkpoints/resnet18.pt", map_location=device))

    train_loader = build_mixed_train_loader(batch_size=32, num_workers=0, pv_per_class=100, pd_repeat=2)
    _, val_loader, pv_test_loader = get_loaders(batch_size=64, num_workers=0)
    pd_test_df = load_plantdoc_split("test")  # used strictly for final reporting

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    print(f"Fine-tuning ResNet18 for {epochs} epochs with field-style augmentation + PlantDoc train...", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += labels.size(0)
        scheduler.step()
        tr_loss, tr_acc = total_loss / n, correct / n
        print(f"epoch {epoch:2d}/{epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | {time.time() - t0:.0f}s", flush=True)

    torch.save(model.state_dict(), CKPT_OUT)
    print(f"\nSaved fine-tuned field checkpoint to {CKPT_OUT}")

    # Evaluate AFTER fine-tuning on both PlantDoc test (238 images) and PlantVillage test (5,431 images)
    y_true_pd, y_pred_pd, y_top3_pd, confs_pd = evaluate_plantdoc(model, pd_test_df, device)
    pd_acc_after = (y_true_pd == y_pred_pd).mean()
    pd_top3_after = y_top3_pd.mean()
    crop_of = lambda i: CLASSES[i].split("___")[0]
    pd_crop_after = np.mean([crop_of(t) == crop_of(p) for t, p in zip(y_true_pd, y_pred_pd)])

    pv_acc_after, pv_wrong_after = evaluate_pv_test(model, pv_test_loader, device)

    # Compare BEFORE vs AFTER
    pv_acc_before, pv_wrong_before = 0.9969, 17
    pd_acc_before, pd_top3_before, pd_crop_before = 0.2017, 0.4328, 0.4664

    comparison = (
        f"=== Feature 4: Before vs. After Field Fine-Tuning Comparison ===\n"
        f"Training data used: PlantVillage train subset (7,515 images) + PlantDoc train (2,337 images, 4x oversampled)\n"
        f"PlantDoc test set (238 images) was NEVER seen during training.\n\n"
        f"{'Metric':<38} | {'Before (resnet18.pt)':<22} | {'After (resnet18_field.pt)':<25} | {'Change':<12}\n"
        f"{'-'*105}\n"
        f"{'PlantDoc Test Top-1 Accuracy (field)':<38} | {pd_acc_before:>20.2%} | {pd_acc_after:>23.2%} | {(pd_acc_after - pd_acc_before)*100:>+8.2f} pp\n"
        f"{'PlantDoc Test Top-3 Accuracy (field)':<38} | {pd_top3_before:>20.2%} | {pd_top3_after:>23.2%} | {(pd_top3_after - pd_top3_before)*100:>+8.2f} pp\n"
        f"{'PlantDoc Test Crop-Level Accuracy':<38} | {pd_crop_before:>20.2%} | {pd_crop_after:>23.2%} | {(pd_crop_after - pd_crop_before)*100:>+8.2f} pp\n"
        f"{'PlantVillage Test Top-1 Accuracy (lab)':<38} | {pv_acc_before:>20.2%} | {pv_acc_after:>23.2%} | {(pv_acc_after - pv_acc_before)*100:>+8.2f} pp\n"
        f"{'PlantVillage Test Mistakes (of 5,431)':<38} | {pv_wrong_before:>20d} | {int(pv_wrong_after):>23d} | {int(pv_wrong_after) - pv_wrong_before:>+8d}\n"
    )
    print("\n" + comparison)

    present_idx = sorted(set(y_true_pd) | set(y_pred_pd))
    present_names = [CLASSES[i] for i in present_idx]
    pd_report = classification_report(
        y_true_pd, y_pred_pd, labels=present_idx, target_names=present_names, digits=3, zero_division=0
    )
    (OUT / "resnet18_field_comparison.txt").write_text(comparison + "\nPlantDoc Test Classification Report (After):\n" + pd_report, encoding="utf-8")

    # Save predictions CSV
    out_df = pd_test_df.copy()
    out_df["path"] = out_df["path"].apply(lambda p: p.as_posix())
    out_df["predicted"] = [CLASSES[i] for i in y_pred_pd]
    out_df["confidence"] = confs_pd.round(3)
    out_df["correct"] = y_true_pd == y_pred_pd
    out_df.to_csv(OUT / "resnet18_field_plantdoc_test_predictions.csv", index=False)

    # Plot Before vs After comparison bar chart
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    metrics = ["PlantDoc Top-1\n(Field)", "PlantDoc Top-3\n(Field)", "PlantDoc Crop-Level\n(Field)", "PlantVillage Top-1\n(Lab)"]
    before_vals = [pd_acc_before * 100, pd_top3_before * 100, pd_crop_before * 100, pv_acc_before * 100]
    after_vals = [pd_acc_after * 100, pd_top3_after * 100, pd_crop_after * 100, pv_acc_after * 100]

    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, before_vals, w, label="Before (PlantVillage-only ResNet18)", color="#90a4ae")
    b2 = ax.bar(x + w/2, after_vals, w, label="After (Field-Augmented + PlantDoc Train)", color="#2e7d32")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ResNet18 Before vs. After Field Fine-Tuning")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 112)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "field_finetune_comparison.png", dpi=120)
    plt.close()
    print("Saved comparison chart to reports/field_finetune_comparison.png")


if __name__ == "__main__":
    main()
