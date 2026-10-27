import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from dataset import get_loaders, CLASSES
from model import SimpleCNN, build_resnet18

BUILDERS = {"simple_cnn": SimpleCNN, "resnet18": build_resnet18}
OUT = Path("reports")

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, preds, confs = [], [], []
    for images, y in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            probs = model(images.to(device)).softmax(1)   # scores -> probabilities
        conf, pred = probs.max(1)                          # top class and how sure it is
        labels += y.tolist(); preds += pred.tolist(); confs += conf.tolist()
    return np.array(labels), np.array(preds), np.array(confs)

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    device = "cuda"
    model = BUILDERS[name]().to(device)
    model.load_state_dict(torch.load(f"checkpoints/{name}.pt", map_location=device))
    _, _, test_loader = get_loaders()
    y, p, conf = predict(model, test_loader, device)

    # 1. Overall accuracy + per-class precision / recall / F1
    acc = (y == p).mean()
    report = classification_report(y, p, target_names=CLASSES, digits=3)
    print(f"test accuracy: {acc:.4f}  ({(y != p).sum()} wrong out of {len(y)})\n")
    print(report)
    (OUT / f"{name}_test_report.txt").write_text(f"test accuracy: {acc:.4f}\n\n{report}")

    # 2. Confusion matrix
    cm = confusion_matrix(y, p)
    fig, ax = plt.subplots(figsize=(18, 18))
    ConfusionMatrixDisplay(cm, display_labels=[c.replace("___", " - ") for c in CLASSES]).plot(
        ax=ax, xticks_rotation=90, colorbar=False, cmap="Greens")
    ax.set_title(f"{name} — test set confusion matrix")
    plt.tight_layout(); plt.savefig(OUT / f"{name}_confusion_matrix.png", dpi=110)

    # 3. List of misclassified images (for the mistake gallery)
    wrong = test_loader.dataset.df[y != p].copy()
    wrong["predicted"] = [CLASSES[i] for i in p[y != p]]
    wrong["confidence"] = conf[y != p].round(3)
    wrong.to_csv(OUT / f"{name}_misclassified.csv", index=False)
    print(f"saved report, confusion matrix and {len(wrong)} misclassified images to reports/")

if __name__ == "__main__":
    main()