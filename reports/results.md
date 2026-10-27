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