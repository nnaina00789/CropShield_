from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path("data/raw/plantvillage dataset/color")
SPLITS = Path("splits")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

CLASSES = sorted(pd.read_csv(SPLITS / "train.csv")["label"].unique())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class PlantDataset(Dataset):
    def __init__(self, split, transform):
        self.df = pd.read_csv(SPLITS / f"{split}.csv")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        path, label = self.df.iloc[i]
        img = Image.open(ROOT / path).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[label]

def get_loaders(batch_size=32, num_workers=4):
    def make(split, tf, shuffle):
        return DataLoader(PlantDataset(split, tf), batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return make("train", train_tf, True), make("val", eval_tf, False), make("test", eval_tf, False)

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()
    images, labels = next(iter(train_loader))
    print(f"{len(CLASSES)} classes")
    print("batches per set:", len(train_loader), len(val_loader), len(test_loader))
    print("image batch shape:", tuple(images.shape))
    print("first 8 labels:", labels[:8].tolist())