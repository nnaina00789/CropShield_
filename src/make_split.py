from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("data/raw/plantvillage dataset/color")
OUT = Path("splits"); OUT.mkdir(exist_ok=True)

rows = [(p.relative_to(ROOT).as_posix(), d.name)
        for d in sorted(ROOT.iterdir()) if d.is_dir()
        for p in d.iterdir()]
df = pd.DataFrame(rows, columns=["path", "label"])

train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:5s} {len(part):6d} images")

check = pd.concat({"train": train.label.value_counts(),
                   "val": val.label.value_counts(),
                   "test": test.label.value_counts()}, axis=1)
print(check.loc[["Potato___healthy", "Orange___Haunglongbing_(Citrus_greening)"]])
