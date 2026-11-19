# CropShield — Results Log

## Model 1: SimpleCNN (baseline, trained from scratch)
- Architecture: 4 conv blocks (Conv-BN-ReLU-MaxPool, 32→256 filters), global avg pool, dropout 0.3, linear (38 classes)
- Parameters: 399,142
- Training: 8 epochs, Adam (lr 0.001), batch size 32, mixed precision, RTX 4050 Laptop GPU
- Time: ~95 s per epoch (~13 min total)
- Best validation accuracy: 96.5% (epoch 7)
- Observation: validation accuracy stayed above training accuracy throughout (augmentation + dropout during training only); no sign of overfitting

## Model 2: ResNet18 (transfer learning, ImageNet-pretrained)
- Architecture: torchvision ResNet18 with ImageNet weights; final layer replaced (512 → 38)
- Parameters: 11,196,006
- Training: 5 epochs, fine-tuning all layers, Adam (lr 0.0003) with cosine annealing, batch size 32, mixed precision
- Time: ~98 s per epoch (~8 min total) — same as SimpleCNN, so data loading (CPU) is the bottleneck, not the GPU
- Best validation accuracy: 99.7% (epoch 5)
- Comparison: SimpleCNN 96.5% → ResNet18 99.7%; validation errors fell from ~190 to ~16 (about 12× fewer)
- Caution: PlantVillage images have plain backgrounds and controlled lighting; real field accuracy is expected to be lower

## Test set evaluation (5,431 unseen images)

| | SimpleCNN | ResNet18 |
|---|---|---|
| Test accuracy | 95.47% | 99.69% |
| Mistakes | 246 | 17 |
| Macro F1 | 0.942 | 0.996 |
| Worst class recall | 72.0% (Tomato Early blight) | 94.1% (Corn Cercospora) |

- ResNet18 test accuracy (99.69%) matches validation (99.7%), so validation was a reliable estimate.
- SimpleCNN shows why overall accuracy misleads: 95.5% overall, but Potato healthy recall 80% and Tomato Early blight recall 72%.
- ResNet18 handles small classes well: Potato healthy (15 test images) 100% correct; macro F1 ≈ weighted F1.

## ResNet18 misclassification analysis (17 mistakes)
- 13 of 17 mistakes are within the same crop: the model recognises the plant reliably; errors are between similar-looking diseases.
- Main confusions: Corn Cercospora → Northern Leaf Blight (3, both cause grey-tan streaks); Tomato Target Spot ↔ Spider mites (5, both cause tiny speckles; the only pest class is mostly confused with a disease).
- Potato Late blight → Tomato Late blight: same pathogen (Phytophthora infestans) on related plants; disease correct, crop wrong.
- Most dangerous error: Tomato Late blight → Tomato healthy (a diseased plant called healthy).
- Data quality issues found: one "Soybean healthy" image is almost entirely background with barely any leaf; one "Cherry healthy" image shows a field of grass, not a single leaf (likely mislabelled). Two of the four cross-crop errors come from faulty images, not model failure.

## Grad-CAM explanations (ResNet18, layer4)
- Method: Grad-CAM on ResNet18's last conv block (512 maps, 7×7), upsampled to 224×224 and overlaid on the image. Implemented from scratch with forward/tensor hooks.
- Correct predictions (diseased leaves): attention falls on lesions and discoloured patches → the model uses disease features, not the background.
- Correct predictions (healthy leaves): attention falls on leaf edges/outline → shape is used to identify the crop; no lesion to focus on.
- Mistakes fall into three types:
  1. Right place, wrong answer — Corn Cercospora → Northern Leaf Blight: attention on a real lesion; the two diseases look alike.
  2. Bad data — "Cherry healthy" image of grass → Strawberry leaf scorch: attention scattered, no single leaf to focus on.
  3. Possible mislabel — Tomato Late blight → healthy (100% confident): leaf shows no visible lesions.
- Limitation: the heatmap starts at 7×7 resolution, so it shows the region the model used, not exact lesion outlines.

## Treatment suggestions
- File: app/remedies.json, covering all 38 classes (verified by src/check_remedies.py).
- Each entry: display name, cause type (fungal / bacterial / viral / pest / water mould / healthy), cause, symptoms, treatment steps, prevention steps.
- Cause type matters for treatment: fungicides do not work on bacteria or viruses; viral diseases (TYLCV, ToMV) and Huanglongbing have no cure, so advice focuses on removal and vector control.
- Safety: guidance names chemical types only, never doses; the app will tell users to confirm with a local agriculture officer / KVK and to follow product labels.
- Sources: TNAU Agritech Portal, UC IPM, APS Disease Lessons.

## Streamlit app (app/app.py)
- Upload a leaf photo → prediction + confidence, Grad-CAM overlay, and treatment advice from remedies.json.
- Reuses project code: eval_tf and CLASSES (dataset.py), build_resnet18 (model.py), GradCAM (gradcam.py); model cached with st.cache_resource.
- Safety: 80% confidence threshold (below it the app shows "Not sure" and no advice) and a disclaimer pointing to the local agriculture officer / KVK.
- Demo tests (images from the test set, in demo_images/):
  - Tomato Late blight → correct, 100% confident; heatmap on the dark lesion.
  - Apple healthy → correct, 100% confident; heatmap over the whole leaf outline.
  - Non-leaf photo (skyscrapers) → best guess "Tomato Leaf mold" at only 34% → app correctly showed "Not sure" with no advice.
- Limitation: the model always picks one of its 38 classes; the threshold catches many unfamiliar images but is not a guaranteed "not a leaf" detector, since models can be confidently wrong.

## Feature 1: App enhancements (camera input, top-3 predictions, Hindi support)
- Camera input: added a photo source toggle between file upload (`st.file_uploader`) and live webcam/phone camera capture (`st.camera_input`).
- Top-3 predictions: updated `GradCAM.__call__` in `src/gradcam.py` to return the full 38-class probability vector alongside the top prediction; `app/app.py` now displays the top 3 classes with `st.progress` probability bars (hidden when confidence is below the 80% threshold so unfamiliar/non-leaf photos aren't given misleading disease labels).
- Hindi language support: added `app/remedies_hi.json` with full Hindi translations for all 38 classes (verified with `src/check_remedies.py`: 38 entries, 0 missing, 0 extra) and a language toggle (`English` / `हिन्दी (Hindi)`) that switches both UI labels and treatment guidance.

## Feature 2: Real-world evaluation on PlantDoc (domain shift analysis)
- Dataset: PlantDoc (`github.com/pratikkayal/PlantDoc-Dataset`) — 2,575 real-world field/in-situ photos (2,337 train, 238 test) across 28 folders (27 represented in the test set; `Tomato two spotted spider mites leaf` has 2 train images and 0 test images).
- Class mapping: every one of PlantDoc's 28 folders maps 1-to-1 onto 28 of our 38 PlantVillage classes (defined in `PLANTDOC_TO_PV` in `src/eval_plantdoc.py`). The remaining 10 PlantVillage classes (e.g., Orange Huanglongbing, Tomato Target Spot, Potato healthy) have no counterpart in PlantDoc.
- Evaluation results on PlantDoc's 238 unseen test images (`src/eval_plantdoc.py`):

| Metric | SimpleCNN | ResNet18 (PlantVillage-trained) |
|---|---|---|
| PlantVillage test accuracy (lab) | 95.47% | 99.69% |
| PlantDoc top-1 accuracy (field) | 10.08% (24/238) | **20.17%** (48/238) |
| PlantDoc top-3 accuracy (field) | 25.63% (61/238) | **43.28%** (103/238) |
| PlantDoc crop-level accuracy | 26.89% | **46.64%** |
| **Accuracy drop (PlantVillage → PlantDoc)** | **−85.39 pp** | **−79.52 pp** |

- Key findings:
  1. **Severe domain shift (~79.5 percentage-point drop for ResNet18):** PlantVillage images show a single detached leaf photographed flat against a uniform grey/black paper background under studio lighting. PlantDoc photos are taken outdoors in fields/gardens with soil, sky, hands, multiple overlapping leaves, fruit, stems, and harsh shadows/glare.
  2. **Over-predicting cluttered/blighted classes:** When faced with busy outdoor backgrounds, the PlantVillage-trained ResNet18 frequently mistakes background clutter and soil for dark necrotic tissue (e.g., `Tomato___Late_blight` recall is 50.0% but precision is only 13.2%, because 38 test images are predicted as Tomato Late blight).
  3. **Classes with distinctive visual textures hold up better:** `Squash___Powdery_mildew` (66.7% recall, 57.1% F1), `Grape___healthy` (53.8% recall, 58.3% F1), and `Raspberry___healthy` (42.9% recall, 54.5% F1) retain noticeably higher accuracy because white powdery patches or distinct leaf serrations remain recognizable in field photos.
  4. **Transfer learning still doubles field accuracy over SimpleCNN:** ResNet18 (20.17% top-1, 43.28% top-3) is twice as accurate as SimpleCNN (10.08% top-1) on real field images, showing that ImageNet pretraining provides more resilient features, though domain adaptation (Feature 4) is clearly needed.

## Feature 3: "Not a leaf" detector (`src/leaf_detector.py`)
- **Problem:** A closed-set 38-class softmax classifier must assign 100% total probability across its 38 leaf categories even when given a selfie, building, vehicle, or household object. Relying only on an 80% disease-confidence threshold is unreliable: on 201 unseen real-world non-leaf photographs, the 38-class ResNet18 assigns $\ge 80\%$ confidence to **30.35% (61/201)** of non-leaf images (false acceptance), while rejecting **47.06% (112/238)** of genuine field leaves from PlantDoc.
- **Architecture:** `LeafDetector` in `src/leaf_detector.py` uses a frozen ImageNet-pretrained ResNet18 backbone (`512`-d global average pooled embedding) with a binary L2-regularised logistic regression head (`checkpoints/leaf_detector.pt`, only 4 KB on disk). Because the ImageNet backbone was trained on 1,000 everyday categories (people, animals, vehicles, indoor objects, plants), its penultimate feature space cleanly separates plant foliage from non-leaf scenes.
- **Training set (1,200 images, strictly from train splits):** 500 PlantVillage train leaves + 500 PlantDoc train field leaves + 200 diverse non-leaf photos.
- **Held-out test set evaluation (739 unseen images: 300 PlantVillage test + 238 PlantDoc test + 201 unseen non-leaf photos):**

| Metric | Naive 80% Disease-Confidence Gate | Dedicated `LeafDetector` (Threshold 0.50) |
|---|---|---|
| **Overall binary accuracy** | 76.73% | **99.05%** (732/739) |
| **ROC-AUC** | — | **0.9998** |
| **PlantVillage leaf recall (lab leaves accepted)** | 99.67% (299/300) | **99.67%** (299/300, mean prob 0.993) |
| **PlantDoc leaf recall (real field leaves accepted)** | 52.94% (126/238) | **99.16%** (236/238, mean prob 0.981) |
| **Non-leaf rejection rate (true negative rate)** | 69.65% (140/201) | **98.01%** (197/201, mean prob 0.023) |
| **Non-leaf false acceptance rate (lower is better)** | 30.35% (61/201) | **1.99%** (4/201) |

- **Webcam selfie verification:** On the user's webcam capture, `LeafDetector` assigns a leaf probability of **0.47%** (< 0.5%), immediately rejecting it with `"Not a plant leaf (0% leaf probability)"` before the disease classifier or Grad-CAM ever runs.
- **Artifacts saved:** `checkpoints/leaf_detector.pt`, `reports/leaf_detector_report.txt`, and `reports/leaf_detector_eval.png` (ROC curve + probability distribution histogram).

## Feature 4: Field fine-tuning (`src/train_field.py` → `checkpoints/resnet18_field.pt`)
- **Methodology:** Started from the PlantVillage-trained `checkpoints/resnet18.pt` and fine-tuned for 6 epochs (`AdamW`, `lr=1.5e-4`, cosine annealing, label smoothing `0.05`, mixed precision) on a balanced mix of **3,800 PlantVillage train images** (100 per class across all 38 classes so no class is forgotten) + **4,674 PlantDoc train images** (2,337 field training images oversampled $2\times$), total 8,474 images/epoch.
- **Field-style augmentation (`field_train_tf`):** Wide-scale random resized crops (`scale=(0.45, 1.0)`), random horizontal/vertical flips, $\pm 35^\circ$ rotation, random perspective distortion (`0.25`, $p=0.4$), strong colour jitter (`brightness=0.35, contrast=0.35, saturation=0.35, hue=0.06`), Gaussian blur ($p=0.25$), and random erasing ($p=0.25$) to simulate outdoor occlusion, shadows, and camera motion.
- **Strict data hygiene:** PlantDoc's 238 test images were **never** loaded during training or hyperparameter selection.
- **Before vs. After comparison on both PlantDoc Test (238 field photos) and PlantVillage Test (5,431 lab photos):**

| Metric | Before (`resnet18.pt`) | After (`resnet18_field.pt`) | Change |
|---|---|---|---|
| **PlantDoc Test Top-1 Accuracy (field)** | 20.17% (48/238) | **61.34%** (146/238) | **+41.17 pp** ($3.04\times$ higher) |
| **PlantDoc Test Top-3 Accuracy (field)** | 43.28% (103/238) | **87.39%** (208/238) | **+44.11 pp** ($2.02\times$ higher) |
| **PlantDoc Test Crop-Level Accuracy** | 46.64% | **79.41%** | **+32.77 pp** |
| **PlantVillage Test Top-1 Accuracy (lab)** | 99.69% (17 mistakes) | **98.47%** (83 mistakes) | **−1.22 pp** |

- **Key takeaways:**
  1. **More than $3\times$ improvement on real field photos:** Top-1 accuracy on unseen PlantDoc test photos jumped from **20.17% → 61.34%** (+41.17 percentage points), and Top-3 accuracy reached **87.39%** (+44.11 pp), meaning the true disease is in the app's Top-3 probability bars nearly 9 times out of 10 on real outdoor photographs.
  2. **Minimal trade-off on studio images:** Because we mixed 100 PlantVillage training images per class into every epoch (replay regularization), accuracy on the 5,431 PlantVillage test images stayed at **98.47%** (only a 1.22 pp trade-off) while retaining full 38-class coverage (including the 10 classes absent from PlantDoc).
  3. **Per-class highlights on PlantDoc test:** `Strawberry___healthy` (100% recall, 0.941 F1), `Raspberry___healthy` (100% recall, 0.933 F1), `Corn___Common_rust_` (90.0% recall, 0.783 F1), `Grape___healthy` (84.6% recall, 0.880 F1), `Squash___Powdery_mildew` (83.3% recall, 0.833 F1), and `Grape___Black_rot` (75.0% recall, 0.800 F1).
  4. **App updated:** `app/app.py` now loads `checkpoints/resnet18_field.pt` automatically.
- **Artifacts saved:** `checkpoints/resnet18_field.pt`, `reports/resnet18_field_comparison.txt`, `reports/resnet18_field_plantdoc_test_predictions.csv`, and `reports/field_finetune_comparison.png`.