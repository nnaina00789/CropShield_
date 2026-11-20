from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from dataset import eval_tf, ROOT as PV_ROOT

CKPT_PATH = Path("checkpoints/leaf_detector.pt")
PLANTDOC_ROOT = Path("data/raw/PlantDoc-Dataset")
NON_LEAF_ROOT = Path("data/raw/non_leaf")
OUT = Path("reports"); OUT.mkdir(exist_ok=True)


class LeafDetector(nn.Module):
    def __init__(self, threshold=0.50):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(512, 1)
        self.threshold = threshold

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def extract_features(self, paths, device="cuda"):
        feats = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = eval_tf(img).unsqueeze(0).to(device)
            feats.append(self.backbone(x)[0].cpu().numpy())
        return np.array(feats, dtype=np.float32)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        feat = self.backbone(x)
        prob = torch.sigmoid(self.head(feat))[0, 0].item()
        return prob >= self.threshold, prob


def load_leaf_detector(device="cuda", threshold=0.50):
    detector = LeafDetector(threshold=threshold).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    detector.head.load_state_dict(state)
    detector.eval()
    return detector


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(42)
    np.random.seed(42)

    detector = LeafDetector().to(device)

    pv_train_df = pd.read_csv("splits/train.csv").sample(500, random_state=42)
    pv_train_paths = [PV_ROOT / p for p in pv_train_df["path"]]

    pd_train_all = [
        p for d in sorted((PLANTDOC_ROOT / "train").iterdir()) if d.is_dir()
        for p in sorted(d.iterdir()) if p.is_file()
    ]
    pd_train_paths = random.sample(pd_train_all, 500)

    non_leaf_all = sorted(NON_LEAF_ROOT.glob("*.jpg"))
    nl_train_paths = non_leaf_all[:200]
    nl_test_paths = non_leaf_all[200:]

    pv_test_df = pd.read_csv("splits/test.csv").sample(300, random_state=42)
    pv_test_paths = [PV_ROOT / p for p in pv_test_df["path"]]

    pd_test_paths = [
        p for d in sorted((PLANTDOC_ROOT / "test").iterdir()) if d.is_dir()
        for p in sorted(d.iterdir()) if p.is_file()
    ]

    print(f"Extracting ImageNet ResNet18 features on {device}...")
    X_pv_tr = detector.extract_features(pv_train_paths, device)
    X_pd_tr = detector.extract_features(pd_train_paths, device)
    X_nl_tr = detector.extract_features(nl_train_paths, device)

    X_pv_te = detector.extract_features(pv_test_paths, device)
    X_pd_te = detector.extract_features(pd_test_paths, device)
    X_nl_te = detector.extract_features(nl_test_paths, device)

    X_tr = np.vstack([X_pv_tr, X_pd_tr, X_nl_tr])
    y_tr = np.array([1] * (len(X_pv_tr) + len(X_pd_tr)) + [0] * len(X_nl_tr))

    clf = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced", random_state=42)
    clf.fit(X_tr, y_tr)

    with torch.no_grad():
        detector.head.weight.copy_(torch.from_numpy(clf.coef_.astype(np.float32)))
        detector.head.bias.copy_(torch.from_numpy(clf.intercept_.astype(np.float32)))
    CKPT_PATH.parent.mkdir(exist_ok=True)
    torch.save(detector.head.state_dict(), CKPT_PATH)
    print(f"Saved leaf detector head weights to {CKPT_PATH} ({CKPT_PATH.stat().st_size:,} bytes)\n")

    X_te = np.vstack([X_pv_te, X_pd_te, X_nl_te])
    y_te = np.array([1] * (len(X_pv_te) + len(X_pd_te)) + [0] * len(X_nl_te))
    probs_te = clf.predict_proba(X_te)[:, 1]
    preds_te = (probs_te >= 0.50).astype(int)

    auc = roc_auc_score(y_te, probs_te)
    overall_acc = (preds_te == y_te).mean()

    pv_probs = clf.predict_proba(X_pv_te)[:, 1]
    pd_probs = clf.predict_proba(X_pd_te)[:, 1]
    nl_probs = clf.predict_proba(X_nl_te)[:, 1]

    pv_acc = (pv_probs >= 0.50).mean()
    pd_acc = (pd_probs >= 0.50).mean()
    nl_acc = (nl_probs < 0.50).mean()

    from model import build_resnet18
    disease_model = build_resnet18().to(device).eval()
    disease_model.load_state_dict(torch.load("checkpoints/resnet18.pt", map_location=device))

    @torch.no_grad()
    def max_disease_conf(paths):
        confs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = eval_tf(img).unsqueeze(0).to(device)
            confs.append(disease_model(x).softmax(1).max().item())
        return np.array(confs)

    nl_disease_confs = max_disease_conf(nl_test_paths)
    pd_disease_confs = max_disease_conf(pd_test_paths)
    pv_disease_confs = max_disease_conf(pv_test_paths)

    false_accept_thresh80 = (nl_disease_confs >= 0.80).mean()
    pd_pass_thresh80 = (pd_disease_confs >= 0.80).mean()
    pv_pass_thresh80 = (pv_disease_confs >= 0.80).mean()

    report = classification_report(
        y_te, preds_te, target_names=["Not a leaf (0)", "Plant leaf (1)"], digits=4
    )

    summary = (
        f"=== Leaf vs. Not-a-Leaf Detector Evaluation ===\n"
        f"Training set : 500 PlantVillage train leaves + 500 PlantDoc train leaves + 200 non-leaf photos (1,200 total)\n"
        f"Test set     : 300 PlantVillage test leaves + 238 PlantDoc test leaves + {len(X_nl_te)} unseen non-leaf photos ({len(y_te)} total)\n\n"
        f"Overall binary accuracy          : {overall_acc:.2%} ({(preds_te == y_te).sum()}/{len(y_te)})\n"
        f"ROC-AUC                          : {auc:.4f}\n"
        f"PlantVillage leaf recall (lab)   : {pv_acc:.2%} ({(pv_probs >= 0.50).sum()}/{len(X_pv_te)}) | mean leaf prob = {pv_probs.mean():.3f}\n"
        f"PlantDoc leaf recall (field)     : {pd_acc:.2%} ({(pd_probs >= 0.50).sum()}/{len(X_pd_te)}) | mean leaf prob = {pd_probs.mean():.3f}\n"
        f"Non-leaf rejection rate (TNR)    : {nl_acc:.2%} ({(nl_probs < 0.50).sum()}/{len(X_nl_te)}) | mean leaf prob = {nl_probs.mean():.3f}\n\n"
        f"--- Comparison: Naive 80% Disease-Confidence Threshold vs. Dedicated Leaf Detector ---\n"
        f"Non-leaf photos falsely accepted (>=80% disease conf) : {false_accept_thresh80:.2%} ({(nl_disease_confs >= 0.80).sum()}/{len(X_nl_te)})\n"
        f"Non-leaf photos falsely accepted by LeafDetector      : {1 - nl_acc:.2%} ({(nl_probs >= 0.50).sum()}/{len(X_nl_te)})\n"
        f"Real field leaves (PlantDoc) accepted by >=80% conf   : {pd_pass_thresh80:.2%} ({(pd_disease_confs >= 0.80).sum()}/{len(X_pd_te)})\n"
        f"Real field leaves (PlantDoc) accepted by LeafDetector : {pd_acc:.2%} ({(pd_probs >= 0.50).sum()}/{len(X_pd_te)})\n\n"
        f"{report}"
    )
    print(summary)
    (OUT / "leaf_detector_report.txt").write_text(summary, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fpr, tpr, _ = roc_curve(y_te, probs_te)
    axes[0].plot(fpr, tpr, color="#2e7d32", lw=2.5, label=f"LeafDetector (AUC = {auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate (Non-leaf accepted as leaf)")
    axes[0].set_ylabel("True Positive Rate (Leaf accepted)")
    axes[0].set_title("ROC Curve — Leaf vs. Not-a-Leaf Detector")
    axes[0].legend(loc="lower right")

    axes[1].hist(nl_probs, bins=25, alpha=0.75, label="Non-leaf photos (test)", color="#d32f2f")
    axes[1].hist(pd_probs, bins=25, alpha=0.75, label="PlantDoc field leaves (test)", color="#1976d2")
    axes[1].hist(pv_probs, bins=25, alpha=0.75, label="PlantVillage lab leaves (test)", color="#388e3c")
    axes[1].axvline(0.50, color="black", linestyle="--", label="Decision threshold (0.50)")
    axes[1].set_xlabel("Predicted Leaf Probability")
    axes[1].set_ylabel("Number of Images")
    axes[1].set_title("Leaf Probability Distribution on Unseen Test Images")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT / "leaf_detector_eval.png", dpi=120)
    plt.close()
    print("Saved evaluation plot to reports/leaf_detector_eval.png")


if __name__ == "__main__":
    main()
