# 🌿 CropShield — Crop Disease & Pest Classifier (SIH26131)

**CropShield** is a supervised deep-learning crop disease and pest diagnosis system built for **Smart India Hackathon Problem Statement SIH26131**. Given a photograph of a crop leaf (uploaded or captured live via webcam/phone camera), CropShield:

1. **Screens out non-leaf photos** with a dedicated **Leaf vs. Not-a-Leaf gatekeeper** (`99.05%` binary accuracy, `0.9998` ROC-AUC) before disease classification runs.
2. **Classifies the leaf across 38 crop-disease categories** covering 14 crops using a field-adapted **ResNet18** transfer learning model (`99.69%` studio test accuracy on PlantVillage; `61.34%` top-1 and `87.39%` top-3 accuracy on real field photos from PlantDoc — a $+41.17\text{ pp}$ improvement over lab-only training).
3. **Explains its prediction visually** with a custom **Grad-CAM** heatmap (`layer4`) highlighting the exact lesion or leaf region that drove the decision.
4. **Provides bilingual treatment & prevention guidance** (**English** and **हिन्दी / Hindi**) tailored to the pathogen type (*fungal, bacterial, viral, water mould, pest, or healthy*), complete with an 80% confidence threshold and agricultural safety disclaimers.

---

## 📊 Summary of Results

### 1. Studio Benchmark: PlantVillage (54,305 images, 38 classes, stratified 80/10/10 split)

| Model | Parameters | Training | Test Accuracy (5,431 unseen) | Mistakes | Macro F1 | Worst Class Recall |
|---|---:|---|---:|---:|---:|---|
| **SimpleCNN** (from scratch) | 399,142 | 8 epochs, Adam (`lr=1e-3`) | 95.47% | 246 | 0.942 | 72.0% (*Tomato Early blight*) |
| **ResNet18** (`resnet18.pt`, ImageNet transfer) | 11,196,006 | 5 epochs, Adam (`lr=3e-4`), cosine annealing | **99.69%** | **17** | **0.996** | **94.1%** (*Corn Cercospora*) |

- **Misclassification insight:** 13 of ResNet18's 17 mistakes on PlantVillage are within the same crop (e.g., *Corn Cercospora* vs. *Northern Leaf Blight*, *Tomato Target Spot* vs. *Spider mites*), and 2 cross-crop errors stem from noisy/mislabelled dataset images.

### 2. Real-World Field Test & Domain Adaptation: PlantDoc (2,575 field images, 28 folders mapped to PlantVillage)

Evaluating models trained solely on studio photos (PlantVillage) against real outdoor photos (PlantDoc's 238 unseen test images) exposes severe lab-to-field domain shift. Fine-tuning with strong field-style augmentation + PlantDoc's training split (`src/train_field.py` → `checkpoints/resnet18_field.pt`, **never** training on PlantDoc test images) more than triples field accuracy while preserving studio performance:

| Metric | SimpleCNN (PV-only) | ResNet18 Before (`resnet18.pt`) | ResNet18 After (`resnet18_field.pt`) | Improvement (Before → After) |
|---|---:|---:|---:|---:|
| **PlantDoc Test Top-1 Accuracy (field)** | 10.08% (24/238) | 20.17% (48/238) | **61.34%** (146/238) | **+41.17 pp** ($3.04\times$) |
| **PlantDoc Test Top-3 Accuracy (field)** | 25.63% (61/238) | 43.28% (103/238) | **87.39%** (208/238) | **+44.11 pp** ($2.02\times$) |
| **PlantDoc Test Crop-Level Accuracy** | 26.89% | 46.64% | **79.41%** | **+32.77 pp** |
| **PlantVillage Test Top-1 Accuracy (lab)** | 95.47% | 99.69% (17 wrong) | **98.47%** (83 wrong) | −1.22 pp |

### 3. "Not a Leaf" Out-of-Distribution Detector (`src/leaf_detector.py`)

A closed-set softmax classifier must assign 100% total probability across its 38 leaf classes even when given a human selfie, building, or vehicle. On 201 unseen non-leaf photographs, a naive 80% disease-confidence threshold falsely accepts **30.35%** of non-leaf images while rejecting **47.06%** of genuine field leaves. Our dedicated `LeafDetector` (frozen ImageNet ResNet18 backbone + L2-regularised logistic regression head, 4 KB checkpoint) solves both problems:

| Metric (739 Unseen Test Images) | Naive 80% Disease-Confidence Gate | Dedicated `LeafDetector` (`checkpoints/leaf_detector.pt`) |
|---|---:|---:|
| **Overall Binary Accuracy** | 76.73% | **99.05%** (732/739) |
| **ROC-AUC** | — | **0.9998** |
| **PlantVillage Leaf Recall (lab leaves accepted)** | 99.67% (299/300) | **99.67%** (299/300) |
| **PlantDoc Leaf Recall (real field leaves accepted)** | 52.94% (126/238) | **99.16%** (236/238) |
| **Non-Leaf Rejection Rate (True Negative Rate)** | 69.65% (140/201) | **98.01%** (197/201) |
| **Non-Leaf False Acceptance Rate** | 30.35% (61/201) | **1.99%** (4/201) |

---

## 🗂️ Project Structure

```text
CropShield_/
├── app/
│   ├── app.py                  # Streamlit web app (upload/camera, LeafDetector, Grad-CAM, top-3 bars, English/Hindi)
│   ├── remedies.json           # English cause, symptoms, treatment & prevention for all 38 classes
│   └── remedies_hi.json        # Hindi (हिन्दी) cause, symptoms, treatment & prevention for all 38 classes
├── src/
│   ├── explore.py              # Dataset class counts, class_distribution.png, samples.png
│   ├── make_split.py           # Stratified 80/10/10 train/val/test split generator
│   ├── dataset.py              # PyTorch Dataset, transforms, DataLoaders, and 38 CLASSES list
│   ├── model.py                # SimpleCNN baseline and build_resnet18 transfer learning architecture
│   ├── train.py                # Mixed-precision training loop with cosine annealing
│   ├── evaluate.py             # Test set accuracy, per-class report, confusion matrix, misclassified CSV
│   ├── mistakes.py             # Same-crop vs cross-crop mistake analysis and mistake gallery plot
│   ├── gradcam.py              # From-scratch Grad-CAM using forward/backward hooks on ResNet18 layer4
│   ├── check_remedies.py       # Verifies remedies.json and remedies_hi.json against all 38 classes
│   ├── make_demo.py            # Exports 1 test image per class into demo_images/
│   ├── eval_plantdoc.py        # Evaluates checkpoints on PlantDoc real field photos (28-class mapping)
│   ├── leaf_detector.py        # Trains & evaluates the Leaf vs. Not-a-Leaf detector (ROC-AUC 0.9998)
│   ├── train_field.py          # Field-style augmentation + PlantDoc train fine-tuning -> resnet18_field.pt
│   └── deploy_hf.py            # Bundles and deploys the app to Hugging Face Spaces
├── splits/
│   ├── train.csv               # 43,444 training image paths & labels
│   ├── val.csv                 # 5,430 validation image paths & labels
│   └── test.csv                # 5,431 test image paths & labels
├── reports/
│   ├── results.md              # Complete chronological log of all experiments and metrics
│   ├── *_test_report.txt       # Per-class precision/recall/F1 reports (PlantVillage & PlantDoc)
│   ├── *_confusion_matrix.png  # 38x38 confusion matrices for SimpleCNN and ResNet18
│   ├── gradcam_correct.png     # Grad-CAM heatmaps on correctly classified test leaves
│   ├── gradcam_mistakes.png    # Grad-CAM heatmaps on top misclassified test leaves
│   ├── leaf_detector_eval.png  # ROC curve & probability distributions for LeafDetector
│   └── field_finetune_comparison.png # Before vs. After bar chart on PlantDoc & PlantVillage
├── demo_images/                # 38 sample test images (one per class) for quick app testing
└── requirements.txt            # Python package dependencies
```

---

## 🚀 Quickstart (Local Setup)

1. **Create and activate the virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   .venv\Scripts\pip install -r requirements.txt
   ```

2. **Run the Streamlit web application:**
   ```powershell
   .venv\Scripts\streamlit run app/app.py
   ```
   - Switch between **English** and **हिन्दी (Hindi)** in the top-right corner.
   - Choose **Upload photo** (try any image in `demo_images/`) or **Take photo with camera**.

3. **Reproduce evaluations & training:**
   ```powershell
   # Verify English and Hindi treatment databases cover all 38 classes
   .venv\Scripts\python src/check_remedies.py

   # Evaluate on PlantVillage test split (5,431 images)
   .venv\Scripts\python src/evaluate.py resnet18

   # Evaluate on PlantDoc real-world field test split (238 images)
   .venv\Scripts\python src/eval_plantdoc.py resnet18 test

   # Train and evaluate the "Not a leaf" detector
   .venv\Scripts\python src/leaf_detector.py

   # Fine-tune for real field photos and compare before vs. after
   .venv\Scripts\python -u src/train_field.py
   ```

---

## 🌐 Deploying to Hugging Face Spaces

Run `src/deploy_hf.py` with a Hugging Face write token (from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):

```powershell
.venv\Scripts\python src/deploy_hf.py --token hf_YOUR_WRITE_TOKEN
```

This script packages `app/`, `src/`, `splits/train.csv`, `demo_images/`, `checkpoints/leaf_detector.pt`, and `checkpoints/resnet18_field.pt` into a self-contained Docker/Streamlit Space and prints your live public URL (`https://huggingface.co/spaces/<username>/CropShield`).

---

## ⚠️ Safety Disclaimer

CropShield is an educational and decision-support tool that provides general guidance only. Chemical categories (never dosages) are mentioned solely for informational purposes. Farmers and users should always confirm diagnoses with a local agriculture officer or **Krishi Vigyan Kendra (KVK)** before applying any treatment and strictly follow product labels.
