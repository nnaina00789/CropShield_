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