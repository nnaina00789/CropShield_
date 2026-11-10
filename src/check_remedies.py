import json
from dataset import CLASSES

remedies = json.load(open("app/remedies.json", encoding="utf-8"))
missing = [c for c in CLASSES if c not in remedies]
extra = [k for k in remedies if k not in CLASSES]
print(f"{len(remedies)} entries | missing: {missing} | extra: {extra}")
print(json.dumps(remedies["Tomato___Late_blight"], indent=2))