from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("data/raw/plantvillage dataset/color")
OUT = Path("splits"); OUT.mkdir(exist_ok=True)

# One row per image: its path (relative to ROOT) and its class label
rows = [(p.relative_to(ROOT).as_posix(), d.name)
        for d in sorted(ROOT.iterdir()) if d.is_dir()
        for p in d.iterdir()]
df = pd.DataFrame(rows, columns=["path", "label"])

# 80% train, 20% temp -> then split temp in half: 10% val, 10% test
train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:5s} {len(part):6d} images")

# Check: smallest and largest class across the three sets
check = pd.concat({"train": train.label.value_counts(),
                   "val": val.label.value_counts(),
                   "test": test.label.value_counts()}, axis=1)
print(check.loc[["Potato___healthy", "Orange___Haunglongbing_(Citrus_greening)"]])