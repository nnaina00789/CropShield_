import json
from dataset import CLASSES

for path in ["app/remedies.json", "app/remedies_hi.json"]:
    remedies = json.load(open(path, encoding="utf-8"))
    missing = [c for c in CLASSES if c not in remedies]
    extra = [k for k in remedies if k not in CLASSES]
    print(f"{path}: {len(remedies)} entries | missing: {missing} | extra: {extra}")
print(json.dumps(remedies["Tomato___Late_blight"], indent=2, ensure_ascii=False))
