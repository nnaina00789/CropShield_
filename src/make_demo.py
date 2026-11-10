import shutil
from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/plantvillage dataset/color")
OUT = Path("demo_images"); OUT.mkdir(exist_ok=True)

test = pd.read_csv("splits/test.csv")
for label, group in test.groupby("label"):
    path = group.iloc[0]["path"]                     # first test image of each class
    shutil.copy(ROOT / path, OUT / f"{label}.jpg")   # name the copy after its class
print(f"copied {len(list(OUT.iterdir()))} images to {OUT}/")