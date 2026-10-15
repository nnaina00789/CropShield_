from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/raw/plantvillage dataset/color")
OUT = Path("reports"); OUT.mkdir(exist_ok=True)

# Count images in each class folder
counts = {d.name: len(list(d.iterdir())) for d in sorted(ROOT.iterdir()) if d.is_dir()}
print(f"{len(counts)} classes, {sum(counts.values())} images\n")
for cls, n in sorted(counts.items(), key=lambda x: x[1]):
    print(f"{n:6d}  {cls}")

# Bar chart of class sizes
plt.figure(figsize=(10, 12))
plt.barh(list(counts), list(counts.values()))
plt.title("Images per class"); plt.tight_layout()
plt.savefig(OUT / "class_distribution.png")

# Grid of 12 random sample images
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax, cls in zip(axes.flat, random.sample(list(counts), 12)):
    img = random.choice(list((ROOT / cls).iterdir()))
    ax.imshow(Image.open(img)); ax.set_title(cls.replace("___", "\n"), fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "samples.png")
print("\nSaved plots to reports/")