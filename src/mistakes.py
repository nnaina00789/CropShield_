import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/raw/plantvillage dataset/color")
name = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
df = pd.read_csv(f"reports/{name}_misclassified.csv").sort_values("confidence", ascending=False)

# 1. Do mistakes stay within the same crop? Which pairs get confused?
crop = lambda c: c.split("___")[0]
same = (df["label"].map(crop) == df["predicted"].map(crop)).sum()
print(f"{len(df)} mistakes, {same} of them within the same crop\n")
print((df["label"] + "  ->  " + df["predicted"]).value_counts().to_string())

# 2. Picture gallery, most confident mistakes first
show = df.head(20)
cols = 5
rows = -(-len(show) // cols)  # round up
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.8))
for ax in axes.flat:
    ax.axis("off")
for ax, r in zip(axes.flat, show.itertuples()):
    ax.imshow(Image.open(ROOT / r.path))
    ax.set_title(f"TRUE: {r.label.replace('___', ' - ')}\n"
                 f"PRED: {r.predicted.replace('___', ' - ')}\n"
                 f"({r.confidence:.0%} confident)", fontsize=7)
plt.tight_layout()
plt.savefig(f"reports/{name}_mistakes.png", dpi=120)
print(f"\nsaved reports/{name}_mistakes.png")