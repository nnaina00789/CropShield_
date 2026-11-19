import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from dataset import eval_tf, CLASSES, CLASS_TO_IDX
from model import SimpleCNN, build_resnet18

PLANTDOC_ROOT = Path("data/raw/PlantDoc-Dataset")
OUT = Path("reports"); OUT.mkdir(exist_ok=True)

# Exact 1-to-1 mapping from PlantDoc's 28 folder names to PlantVillage's 38 class names
PLANTDOC_TO_PV = {
    "Apple Scab Leaf":                      "Apple___Apple_scab",
    "Apple leaf":                           "Apple___healthy",
    "Apple rust leaf":                      "Apple___Cedar_apple_rust",
    "Bell_pepper leaf":                     "Pepper,_bell___healthy",
    "Bell_pepper leaf spot":                "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf":                       "Blueberry___healthy",
    "Cherry leaf":                          "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot":                  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight":                     "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf":                       "Corn_(maize)___Common_rust_",
    "Peach leaf":                           "Peach___healthy",
    "Potato leaf early blight":             "Potato___Early_blight",
    "Potato leaf late blight":              "Potato___Late_blight",
    "Raspberry leaf":                       "Raspberry___healthy",
    "Soyabean leaf":                        "Soybean___healthy",
    "Squash Powdery mildew leaf":           "Squash___Powdery_mildew",
    "Strawberry leaf":                      "Strawberry___healthy",
    "Tomato Early blight leaf":             "Tomato___Early_blight",
    "Tomato Septoria leaf spot":            "Tomato___Septoria_leaf_spot",
    "Tomato leaf":                          "Tomato___healthy",
    "Tomato leaf bacterial spot":           "Tomato___Bacterial_spot",
    "Tomato leaf late blight":              "Tomato___Late_blight",
    "Tomato leaf mosaic virus":             "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus":             "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf":                     "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf":                           "Grape___healthy",
    "grape leaf black rot":                 "Grape___Black_rot",
}

def load_plantdoc_split(split="test"):
    rows = []
    split_dir = PLANTDOC_ROOT / split
    for d in sorted(split_dir.iterdir()):
        if not d.is_dir() or d.name not in PLANTDOC_TO_PV:
            continue
        pv_label = PLANTDOC_TO_PV[d.name]
        for p in sorted(d.iterdir()):
            if p.is_file():
                rows.append((p, d.name, pv_label, CLASS_TO_IDX[pv_label]))
    return pd.DataFrame(rows, columns=["path", "plantdoc_label", "label", "idx"])

@torch.no_grad()
def evaluate_plantdoc(model, df, device="cuda"):
    model.eval()
    y_true, y_pred, y_top3, confs = [], [], [], []
    for r in df.itertuples():
        img = Image.open(r.path).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(device)
        probs = model(x).softmax(1)[0].cpu().numpy()
        top3 = np.argsort(probs)[::-1][:3]
        pred = int(top3[0])
        conf = float(probs[pred])
        y_true.append(r.idx)
        y_pred.append(pred)
        y_top3.append(r.idx in top3)
        confs.append(conf)
    return np.array(y_true), np.array(y_pred), np.array(y_top3), np.array(confs)

def main():
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    builder = SimpleCNN if ckpt_name == "simple_cnn" else build_resnet18
    model = builder().to(device)
    model.load_state_dict(torch.load(f"checkpoints/{ckpt_name}.pt", map_location=device))

    df = load_plantdoc_split(split)
    y_true, y_pred, y_top3, confs = evaluate_plantdoc(model, df, device)

    acc = (y_true == y_pred).mean()
    top3_acc = y_top3.mean()
    crop_of = lambda i: CLASSES[i].split("___")[0]
    crop_acc = np.mean([crop_of(t) == crop_of(p) for t, p in zip(y_true, y_pred)])

    present_idx = sorted(set(y_true) | set(y_pred))
    present_names = [CLASSES[i] for i in present_idx]
    report = classification_report(
        y_true, y_pred, labels=present_idx, target_names=present_names, digits=3, zero_division=0
    )

    pv_acc = 0.9969 if ckpt_name == "resnet18" else 0.9547
    summary = (
        f"Model: {ckpt_name} | Split: PlantDoc {split} ({len(df)} images across {df['label'].nunique()} mapped classes)\n"
        f"PlantVillage test accuracy : {pv_acc:.2%}\n"
        f"PlantDoc top-1 accuracy    : {acc:.2%} ({(y_true == y_pred).sum()}/{len(df)} correct)\n"
        f"PlantDoc top-3 accuracy    : {top3_acc:.2%} ({y_top3.sum()}/{len(df)} in top-3)\n"
        f"PlantDoc crop-level acc    : {crop_acc:.2%} (right plant, any disease/healthy state)\n"
        f"Accuracy drop (PV -> PD)   : {(pv_acc - acc) * 100:.2f} percentage points\n"
    )
    print(summary)
    print(report)

    tag = f"{ckpt_name}_plantdoc_{split}"
    (OUT / f"{tag}_report.txt").write_text(summary + "\n" + report, encoding="utf-8")

    # Save predictions CSV
    out_df = df.copy()
    out_df["path"] = out_df["path"].apply(lambda p: p.relative_to(PLANTDOC_ROOT).as_posix())
    out_df["predicted"] = [CLASSES[i] for i in y_pred]
    out_df["confidence"] = confs.round(3)
    out_df["correct"] = y_true == y_pred
    out_df.to_csv(OUT / f"{tag}_predictions.csv", index=False)

    # Save side-by-side comparison bar chart for test split
    if split == "test" and ckpt_name == "resnet18":
        per_cls = out_df.groupby("label")["correct"].agg(["count", "mean"]).sort_values("mean")
        fig, ax = plt.subplots(figsize=(10, 8))
        labels_clean = [c.replace("___", " - ") for c in per_cls.index]
        ax.barh(labels_clean, per_cls["mean"] * 100, color="#2e7d32")
        ax.axvline(acc * 100, color="red", linestyle="--", label=f"Overall PlantDoc top-1 ({acc:.1%})")
        ax.set_xlabel("Test Recall / Accuracy (%)")
        ax.set_title("ResNet18 (PlantVillage-trained) on PlantDoc Real Field Photos (238 test images)")
        ax.set_xlim(0, 105)
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT / "plantdoc_baseline_per_class.png", dpi=120)
        plt.close()

if __name__ == "__main__":
    main()
