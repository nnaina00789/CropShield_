from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from dataset import eval_tf, CLASSES, ROOT
from model import build_resnet18

class GradCAM:
    def __init__(self, model, layer):
        self.model = model.eval()
        layer.register_forward_hook(self._catch)   # runs every time data passes through `layer`

    def _catch(self, module, inputs, output):
        self.maps = output                                           # the 512 pattern maps (7x7)
        output.register_hook(lambda g: setattr(self, "grads", g))   # their gradients, on backward

    def __call__(self, x):
        logits = self.model(x)
        probs = logits.softmax(1)[0]
        idx = probs.argmax().item()                                  # predicted class
        self.model.zero_grad()
        logits[0, idx].backward()                                    # gradients of that class's score
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # 1. importance of each map
        cam = F.relu((weights * self.maps).sum(1, keepdim=True))     # 2-3. weighted sum, keep positive
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]  # 4. 7x7 -> 224x224
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)     # scale to 0..1 for colouring
        return cam.detach().cpu().numpy(), idx, probs[idx].item()

def explain(gradcam, rows, out_file):
    fig, axes = plt.subplots(len(rows), 2, figsize=(7, 3.4 * len(rows)))
    for ax_row, r in zip(axes, rows.itertuples()):
        img = Image.open(ROOT / r.path).convert("RGB")
        heat, idx, conf = gradcam(eval_tf(img).unsqueeze(0).to("cuda"))
        small = img.resize((224, 224))
        ax_row[0].imshow(small)
        ax_row[0].set_title(f"TRUE: {r.label.replace('___', ' - ')}", fontsize=9)
        ax_row[1].imshow(small)
        ax_row[1].imshow(heat, cmap="jet", alpha=0.45)               # heatmap on top of the photo
        ax_row[1].set_title(f"PRED: {CLASSES[idx].replace('___', ' - ')} ({conf:.0%})", fontsize=9)
        for a in ax_row:
            a.axis("off")
    plt.tight_layout(); plt.savefig(out_file, dpi=110); plt.close()
    print("saved", out_file)

def main():
    model = build_resnet18().to("cuda")
    model.load_state_dict(torch.load("checkpoints/resnet18.pt", map_location="cuda"))
    gradcam = GradCAM(model, model.layer4)

    test = pd.read_csv("splits/test.csv")
    wrong = pd.read_csv("reports/resnet18_misclassified.csv")
    # 8 correctly classified images, each from a different class
    correct = (test[~test["path"].isin(wrong["path"])]
               .groupby("label").sample(1, random_state=0).sample(8, random_state=0))
    explain(gradcam, correct, "reports/gradcam_correct.png")
    # the 8 most confident mistakes
    explain(gradcam, wrong.sort_values("confidence", ascending=False).head(8), "reports/gradcam_mistakes.png")

if __name__ == "__main__":
    main()